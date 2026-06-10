import importlib

import torch
import torch.nn.functional as F

_WARNED_FALLBACKS = set()


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
