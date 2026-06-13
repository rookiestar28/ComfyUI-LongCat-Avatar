from __future__ import annotations

import importlib
import math
import re
from typing import Any

import torch
import torch.nn.functional as F

_WARNED_FALLBACKS = set()
_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


_DTYPE_BYTE_SIZES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}


def callable_or_none(module_name, attr_name):
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    candidate = getattr(module, attr_name, None)
    return candidate if callable(candidate) else None


def warn_attention_fallback(backend, reason, fallback="SDPA"):
    key = (backend, reason, fallback)
    if key in _WARNED_FALLBACKS:
        return
    _WARNED_FALLBACKS.add(key)
    print(f"[WARN] LongCat attention backend '{backend}' unavailable: {reason}; falling back to {fallback}.")


def _dtype_num_bytes(dtype: Any) -> int:
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    if dtype in _DTYPE_BYTE_SIZES:
        return _DTYPE_BYTE_SIZES[dtype]
    raise ValueError(f"Unsupported attention score dtype for byte estimation: {dtype!r}")


def _device_type(tensor) -> str:
    device_type = getattr(getattr(tensor, "device", None), "type", None)
    if device_type:
        return str(device_type)
    return str(getattr(tensor, "device", "cpu")).split(":", 1)[0]


def _shape_tuple(tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(tensor, "shape", ()))


def _validate_qkv(q, k, v) -> tuple[int, int, int, int, int, int]:
    q_shape = _shape_tuple(q)
    k_shape = _shape_tuple(k)
    v_shape = _shape_tuple(v)
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        raise ValueError(
            "LongCat attention helpers require q, k, and v shaped [B, H, S, D]; "
            f"got q={q_shape}, k={k_shape}, v={v_shape}."
        )

    bq, hq, sq, dq = q_shape
    bk, hk, sk, dk = k_shape
    bv, hv, sv, dv = v_shape
    if (bq, hq) != (bk, hk) or (bq, hq) != (bv, hv):
        raise ValueError(
            "LongCat attention helpers require matching batch/head dimensions; "
            f"got q={q_shape}, k={k_shape}, v={v_shape}."
        )
    if dq != dk:
        raise ValueError(f"LongCat attention q/k head dimensions must match; got q={dq}, k={dk}.")
    if sk != sv:
        raise ValueError(f"LongCat attention k/v sequence lengths must match; got k={sk}, v={sv}.")
    return bq, hq, sq, sk, dq, dv


def attention_score_buffer_bytes(q, k, chunk_size=None, score_dtype=None) -> int:
    """Return estimated bytes for the dense attention score buffer."""

    bsz, heads, query_tokens, key_tokens, _, _ = _validate_qkv(q, k, k)
    if chunk_size is not None:
        if int(chunk_size) < 1:
            raise ValueError(f"chunk_size must be positive when provided; got {chunk_size!r}.")
        query_tokens = min(query_tokens, int(chunk_size))
    dtype = score_dtype if score_dtype is not None else getattr(q, "dtype", torch.float32)
    return int(bsz) * int(heads) * int(query_tokens) * int(key_tokens) * _dtype_num_bytes(dtype)


def select_query_chunk_size(q, k, max_score_bytes: int, score_dtype=None) -> int:
    """Choose a query chunk size that keeps one score buffer under max_score_bytes."""

    bsz, heads, query_tokens, key_tokens, _, _ = _validate_qkv(q, k, k)
    if query_tokens == 0:
        return 0
    budget = int(max_score_bytes)
    if budget < 1:
        raise ValueError(f"max_score_bytes must be positive; got {max_score_bytes!r}.")
    dtype = score_dtype if score_dtype is not None else getattr(q, "dtype", torch.float32)
    bytes_per_query_token = int(bsz) * int(heads) * int(key_tokens) * _dtype_num_bytes(dtype)
    if bytes_per_query_token <= 0:
        return int(query_tokens)
    return max(1, min(int(query_tokens), budget // bytes_per_query_token))


def chunked_eager_attention(q, k, v, attn_mask=None, chunk_size=None, max_score_bytes=None, scale=None):
    """
    Exact eager attention for [B, H, S, D] tensors using query chunks.

    IMPORTANT: do not add key/window chunking here without an online-softmax design; restricting K/V changes attention
    semantics and would silently alter Avatar DiT behavior.
    """

    if attn_mask is not None:
        raise NotImplementedError("chunked_eager_attention currently supports only unmasked attention.")

    bsz, heads, query_tokens, key_tokens, head_dim, value_dim = _validate_qkv(q, k, v)
    if query_tokens == 0:
        return torch.empty((bsz, heads, 0, value_dim), dtype=v.dtype, device=v.device)
    if key_tokens == 0:
        raise ValueError("chunked_eager_attention requires at least one key/value token.")

    if chunk_size is None:
        if max_score_bytes is None:
            chunk_size = query_tokens
        else:
            chunk_size = select_query_chunk_size(q, k, int(max_score_bytes))
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive; got {chunk_size!r}.")

    q_work = q.contiguous()
    k_work = k.contiguous()
    v_work = v.contiguous()
    output = torch.empty((bsz, heads, query_tokens, value_dim), dtype=v.dtype, device=v.device)
    softmax_scale = float(scale) if scale is not None else float(head_dim) ** -0.5
    k_transposed = k_work.transpose(-2, -1)

    for start in range(0, query_tokens, chunk_size):
        end = min(start + chunk_size, query_tokens)
        scores = torch.matmul(q_work[:, :, start:end, :], k_transposed)
        if not math.isclose(softmax_scale, 1.0):
            scores = scores * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        output[:, :, start:end, :] = torch.matmul(probs, v_work)
    return output


def _safe_label(value) -> str:
    label = str(value or "attention")
    return _SAFE_LABEL_RE.sub("_", label)[:80]


def _mps_memory_summary(torch_module) -> dict[str, int | None]:
    mps = getattr(torch_module, "mps", None)
    result: dict[str, int | None] = {}
    for field, api_name in (
        ("mps_allocated_bytes", "current_allocated_memory"),
        ("mps_driver_allocated_bytes", "driver_allocated_memory"),
        ("mps_recommended_max_bytes", "recommended_max_memory"),
    ):
        fn = getattr(mps, api_name, None)
        if not callable(fn):
            result[field] = None
            continue
        try:
            result[field] = int(fn())
        except Exception:
            result[field] = None
    return result


def attention_telemetry_summary(
    label,
    q,
    k,
    v,
    *,
    chunk_size=None,
    max_score_bytes=None,
    score_dtype=None,
    torch_module=torch,
) -> dict[str, Any]:
    full_score_bytes = attention_score_buffer_bytes(q, k, score_dtype=score_dtype)
    selected_chunk = int(chunk_size) if chunk_size is not None else None
    if selected_chunk is None and max_score_bytes is not None:
        selected_chunk = select_query_chunk_size(q, k, int(max_score_bytes), score_dtype=score_dtype)
    chunk_score_bytes = (
        attention_score_buffer_bytes(q, k, chunk_size=selected_chunk, score_dtype=score_dtype)
        if selected_chunk is not None
        else None
    )
    summary: dict[str, Any] = {
        "label": _safe_label(label),
        "device": str(getattr(q, "device", "cpu")),
        "device_type": _device_type(q),
        "q_shape": _shape_tuple(q),
        "k_shape": _shape_tuple(k),
        "v_shape": _shape_tuple(v),
        "q_dtype": str(getattr(q, "dtype", "")),
        "k_dtype": str(getattr(k, "dtype", "")),
        "v_dtype": str(getattr(v, "dtype", "")),
        "q_contiguous": bool(q.is_contiguous()),
        "k_contiguous": bool(k.is_contiguous()),
        "v_contiguous": bool(v.is_contiguous()),
        "score_buffer_bytes": full_score_bytes,
        "chunk_size": selected_chunk,
        "chunk_score_buffer_bytes": chunk_score_bytes,
    }
    if summary["device_type"] == "mps":
        summary.update(_mps_memory_summary(torch_module))
    return summary


def format_attention_telemetry(summary: dict[str, Any]) -> str:
    ordered_fields = (
        "label",
        "device_type",
        "q_shape",
        "k_shape",
        "v_shape",
        "q_dtype",
        "score_buffer_bytes",
        "chunk_size",
        "chunk_score_buffer_bytes",
        "q_contiguous",
        "k_contiguous",
        "v_contiguous",
        "mps_allocated_bytes",
        "mps_driver_allocated_bytes",
        "mps_recommended_max_bytes",
    )
    fields = [f"{field}={summary[field]}" for field in ordered_fields if field in summary]
    return "[INFO] LongCat attention telemetry " + " ".join(fields)


def sdpa_attention(q, k, v, attn_mask=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)


def sage_attention(q, k, v):
    sageattn = callable_or_none("sageattention", "sageattn")
    if sageattn is None:
        sageattn = callable_or_none("sageattn", "sageattn")
    if sageattn is None:
        warn_attention_fallback("sageattn", "missing sageattention.sageattn")
        return None
    if not (q.dtype == k.dtype == v.dtype):
        return sageattn(q, k.to(q.dtype), v.to(q.dtype), dropout_p=0.0, is_causal=False, tensor_layout="HND")
    if q.dtype == torch.float32:
        return sageattn(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
            dropout_p=0.0,
            is_causal=False,
            tensor_layout="HND",
        ).to(torch.float32)
    return sageattn(q, k, v, dropout_p=0.0, is_causal=False, tensor_layout="HND")


def sage_attention_3(q, k, v):
    sageattn_blackwell = callable_or_none("sageattn3", "sageattn3_blackwell")
    if sageattn_blackwell is None:
        sageattn_blackwell = callable_or_none("sageattention", "sageattn_blackwell")
    if sageattn_blackwell is None:
        sageattn_blackwell = callable_or_none("sageattn", "sageattn_blackwell")
    if sageattn_blackwell is None:
        warn_attention_fallback("sageattn_3", "missing Blackwell SageAttention3 callable")
        return None
    return sageattn_blackwell(q, k, v, per_block_mean=False)


def sage_varlen_attention(q, k, v, query_seqlen, kv_seqlen):
    sageattn_varlen = callable_or_none("sageattention", "sageattn_varlen")
    if sageattn_varlen is None:
        sageattn_varlen = callable_or_none("sageattn", "sageattn_varlen")
    if sageattn_varlen is None:
        warn_attention_fallback("sageattn_varlen", "missing sageattention.sageattn_varlen")
        return None

    cu_seqlens_q = torch.tensor([0] + [query_seqlen] * len(kv_seqlen), device=q.device).cumsum(0).to(torch.int32)
    cu_seqlens_k = torch.tensor([0] + kv_seqlen, device=q.device).cumsum(0).to(torch.int32)
    max_seqlen_k = max(kv_seqlen)
    q_flat, k_flat, v_flat = q[0], k[0], v[0]
    compute_dtype = torch.float16 if q_flat.dtype == torch.float32 else q_flat.dtype
    q_run = q_flat.to(compute_dtype)
    k_run = k_flat.to(compute_dtype)
    v_run = v_flat.to(compute_dtype)
    restore_float32 = q_flat.dtype == torch.float32
    if not (q_flat.dtype == k_flat.dtype == v_flat.dtype):
        output = sageattn_varlen(
            q_run,
            k_run,
            v_run,
            cu_seqlens_q,
            cu_seqlens_k,
            query_seqlen,
            max_seqlen_k,
            dropout_p=0.0,
            is_causal=False,
        )
        return output.to(torch.float32) if restore_float32 else output
    output = sageattn_varlen(
        q_run,
        k_run,
        v_run,
        cu_seqlens_q,
        cu_seqlens_k,
        query_seqlen,
        max_seqlen_k,
        dropout_p=0.0,
        is_causal=False,
    )
    return output.to(torch.float32) if restore_float32 else output
