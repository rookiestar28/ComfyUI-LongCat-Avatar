from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from LongCat_Video.backend_capabilities import mps_cpu_fallback_enabled


@dataclass(frozen=True)
class AttentionProbeSpec:
    device: str = "mps"
    attention_backend: str = "sdpa"
    batch: int = 1
    heads: int = 8
    query_tokens: int = 256
    key_tokens: int = 256
    head_dim: int = 64
    dtype: str = "fp16"
    atol: float = 0.05
    rtol: float = 0.05


@dataclass(frozen=True)
class AttentionProbeResult:
    spec: AttentionProbeSpec
    status: str
    native_mps: bool
    cpu_fallback_enabled: bool
    max_abs_diff: float | None
    reason: str


def build_representative_attention_probe_spec() -> AttentionProbeSpec:
    return AttentionProbeSpec()


def run_mps_attention_equivalence_probe(
    spec: AttentionProbeSpec | None = None,
    *,
    torch_module: Any = None,
    environ: Mapping[str, str] | None = None,
) -> AttentionProbeResult:
    spec = spec or build_representative_attention_probe_spec()
    fallback_enabled = mps_cpu_fallback_enabled(environ or os.environ)
    if fallback_enabled:
        return AttentionProbeResult(
            spec,
            "skipped",
            False,
            True,
            None,
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; native MPS evidence requires fallback disabled.",
        )

    if torch_module is None:
        try:
            import torch as torch_module
        except ModuleNotFoundError:
            return AttentionProbeResult(spec, "skipped", False, False, None, "PyTorch is not installed.")

    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    is_available = getattr(mps_backend, "is_available", None)
    if not callable(is_available) or not bool(is_available()):
        return AttentionProbeResult(spec, "skipped", False, False, None, "MPS backend is not available.")

    try:
        from torch.nn import functional as functional_attention
    except Exception as exc:
        return AttentionProbeResult(
            spec,
            "skipped",
            False,
            False,
            None,
            f"scaled_dot_product_attention import failed: {type(exc).__name__}: {exc}",
        )

    dtype = _torch_dtype(spec.dtype, torch_module)
    shape_q = (spec.batch, spec.heads, spec.query_tokens, spec.head_dim)
    shape_kv = (spec.batch, spec.heads, spec.key_tokens, spec.head_dim)
    try:
        generator = torch_module.Generator(device="cpu").manual_seed(0)
        q_cpu = torch_module.randn(shape_q, generator=generator, device="cpu", dtype=torch_module.float32)
        k_cpu = torch_module.randn(shape_kv, generator=generator, device="cpu", dtype=torch_module.float32)
        v_cpu = torch_module.randn(shape_kv, generator=generator, device="cpu", dtype=torch_module.float32)
        cpu_out = functional_attention.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, dropout_p=0.0)
        q_mps = q_cpu.to(spec.device, dtype=dtype)
        k_mps = k_cpu.to(spec.device, dtype=dtype)
        v_mps = v_cpu.to(spec.device, dtype=dtype)
        mps_out = functional_attention.scaled_dot_product_attention(q_mps, k_mps, v_mps, dropout_p=0.0)
        mps_cpu = mps_out.to("cpu", dtype=torch_module.float32)
        max_abs_diff = float((cpu_out - mps_cpu).abs().max().item())
        passed = bool(torch_module.allclose(cpu_out, mps_cpu, atol=spec.atol, rtol=spec.rtol))
    except Exception as exc:
        return AttentionProbeResult(spec, "fail", True, False, None, f"{type(exc).__name__}: {exc}")

    return AttentionProbeResult(
        spec,
        "pass" if passed else "fail",
        True,
        False,
        max_abs_diff,
        "ok" if passed else "CPU and MPS SDPA outputs exceeded tolerance.",
    )


def _torch_dtype(dtype_name: str, torch_module: Any) -> Any:
    normalized = str(dtype_name).lower()
    if normalized in {"fp16", "float16", "half"}:
        return getattr(torch_module, "float16")
    if normalized in {"bf16", "bfloat16"}:
        return getattr(torch_module, "bfloat16")
    return getattr(torch_module, "float32")
