from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from LongCat_Video.backend_capabilities import BFloat16ProbeResult, normalize_backend_type, probe_bfloat16_support

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised by repo tests without torch installed.
    # CRITICAL: keep torch optional at import time; CI contract tests run without PyTorch installed.
    torch = None


PRECISION_AUTO = "auto"
PRECISION_BF16 = "bf16"
PRECISION_FP16 = "fp16"
PRECISION_FP32 = "fp32"
SUPPORTED_PRECISIONS = (PRECISION_AUTO, PRECISION_BF16, PRECISION_FP16, PRECISION_FP32)


@dataclass(frozen=True)
class BackendDTypePolicy:
    backend: str
    requested_precision: str
    text_encoder_precision: str
    audio_encoder_precision: str
    dit_precision: str
    vae_precision: str
    math_precision: str
    text_encoder_dtype: Any
    audio_encoder_dtype: Any
    dit_dtype: Any
    vae_dtype: Any
    math_dtype: Any
    bfloat16_probe: BFloat16ProbeResult | None
    reason: str


def _normalize_precision(value: Any) -> str:
    normalized = str(value or PRECISION_AUTO).lower()
    aliases = {
        "float16": PRECISION_FP16,
        "half": PRECISION_FP16,
        "float32": PRECISION_FP32,
        "single": PRECISION_FP32,
        "bfloat16": PRECISION_BF16,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported precision '{value}'. Supported values: {', '.join(SUPPORTED_PRECISIONS)}."
        )
    return normalized


def dtype_for_precision(precision: str, *, torch_module: Any = torch) -> Any:
    precision = _normalize_precision(precision)
    attr = {
        PRECISION_BF16: "bfloat16",
        PRECISION_FP16: "float16",
        PRECISION_FP32: "float32",
    }.get(precision)
    if attr is None:
        raise ValueError("auto precision must be resolved before requesting a dtype object.")
    if torch_module is None:
        return precision
    return getattr(torch_module, attr, precision)


def resolve_backend_dtype_policy(
    device: Any,
    *,
    requested_precision: Any = PRECISION_AUTO,
    torch_module: Any = torch,
) -> BackendDTypePolicy:
    backend = normalize_backend_type(device)
    requested = _normalize_precision(requested_precision)
    probe: BFloat16ProbeResult | None = None

    if backend == "cuda":
        if requested not in (PRECISION_AUTO, PRECISION_BF16):
            raise ValueError("CUDA LongCat runtime precision remains bf16-only in this branch.")
        return _build_policy(
            backend=backend,
            requested_precision=requested,
            text_encoder_precision=PRECISION_BF16,
            audio_encoder_precision=PRECISION_BF16,
            dit_precision=PRECISION_BF16,
            vae_precision=PRECISION_BF16,
            math_precision=PRECISION_FP32,
            torch_module=torch_module,
            bfloat16_probe=None,
            reason="CUDA keeps the existing official bf16 runtime precision.",
        )

    if backend == "mps":
        if requested == PRECISION_AUTO:
            return _build_policy(
                backend=backend,
                requested_precision=requested,
                text_encoder_precision=PRECISION_FP16,
                audio_encoder_precision=PRECISION_FP32,
                dit_precision=PRECISION_FP16,
                vae_precision=PRECISION_FP16,
                math_precision=PRECISION_FP32,
                torch_module=torch_module,
                bfloat16_probe=None,
                reason="MPS auto precision uses fp16 model tensors with fp32 audio/math boundaries.",
            )
        if requested == PRECISION_BF16:
            probe = probe_bfloat16_support(device, torch_module=torch_module)
            if not probe.supported:
                raise ValueError(f"MPS bf16 requested but runtime probe failed: {probe.detail}")
            return _build_policy(
                backend=backend,
                requested_precision=requested,
                text_encoder_precision=PRECISION_BF16,
                audio_encoder_precision=PRECISION_FP32,
                dit_precision=PRECISION_BF16,
                vae_precision=PRECISION_BF16,
                math_precision=PRECISION_FP32,
                torch_module=torch_module,
                bfloat16_probe=probe,
                reason="MPS bf16 was explicitly requested and the runtime probe passed.",
            )
        if requested == PRECISION_FP16:
            return _build_policy(
                backend=backend,
                requested_precision=requested,
                text_encoder_precision=PRECISION_FP16,
                audio_encoder_precision=PRECISION_FP32,
                dit_precision=PRECISION_FP16,
                vae_precision=PRECISION_FP16,
                math_precision=PRECISION_FP32,
                torch_module=torch_module,
                bfloat16_probe=None,
                reason="MPS fp16 was explicitly requested.",
            )
        return _build_policy(
            backend=backend,
            requested_precision=requested,
            text_encoder_precision=PRECISION_FP32,
            audio_encoder_precision=PRECISION_FP32,
            dit_precision=PRECISION_FP32,
            vae_precision=PRECISION_FP32,
            math_precision=PRECISION_FP32,
            torch_module=torch_module,
            bfloat16_probe=None,
            reason="MPS fp32 was explicitly requested.",
        )

    if requested not in (PRECISION_AUTO, PRECISION_FP32):
        raise ValueError(f"{backend} LongCat runtime precision is fp32-only.")
    return _build_policy(
        backend=backend,
        requested_precision=requested,
        text_encoder_precision=PRECISION_FP32,
        audio_encoder_precision=PRECISION_FP32,
        dit_precision=PRECISION_FP32,
        vae_precision=PRECISION_FP32,
        math_precision=PRECISION_FP32,
        torch_module=torch_module,
        bfloat16_probe=None,
        reason=f"{backend} backend uses fp32 diagnostics only.",
    )


def _build_policy(
    *,
    backend: str,
    requested_precision: str,
    text_encoder_precision: str,
    audio_encoder_precision: str,
    dit_precision: str,
    vae_precision: str,
    math_precision: str,
    torch_module: Any,
    bfloat16_probe: BFloat16ProbeResult | None,
    reason: str,
) -> BackendDTypePolicy:
    return BackendDTypePolicy(
        backend=backend,
        requested_precision=requested_precision,
        text_encoder_precision=text_encoder_precision,
        audio_encoder_precision=audio_encoder_precision,
        dit_precision=dit_precision,
        vae_precision=vae_precision,
        math_precision=math_precision,
        text_encoder_dtype=dtype_for_precision(text_encoder_precision, torch_module=torch_module),
        audio_encoder_dtype=dtype_for_precision(audio_encoder_precision, torch_module=torch_module),
        dit_dtype=dtype_for_precision(dit_precision, torch_module=torch_module),
        vae_dtype=dtype_for_precision(vae_precision, torch_module=torch_module),
        math_dtype=dtype_for_precision(math_precision, torch_module=torch_module),
        bfloat16_probe=bfloat16_probe,
        reason=reason,
    )


def mps_safe_numeric_dtype_name(device: Any, dtype_name: Any) -> str:
    normalized = str(dtype_name).lower().replace("torch.", "")
    if normalize_backend_type(device) != "mps":
        return normalized
    if normalized in {"float64", "double"}:
        return "float32"
    if normalized in {"int64", "long"}:
        return "int32"
    return normalized


def resolve_random_generator_device(device: Any) -> str:
    if normalize_backend_type(device) == "mps":
        return "cpu"
    return str(device or "cpu").lower()


def randn_for_device(
    shape: Any,
    *,
    device: Any,
    dtype: Any,
    generator: Any = None,
    torch_module: Any = torch,
) -> Any:
    if torch_module is None:
        raise RuntimeError("PyTorch is required for random tensor generation.")
    if normalize_backend_type(device) == "mps":
        cpu_latents = torch_module.randn(shape, generator=generator, device="cpu", dtype=dtype)
        return cpu_latents.to(device=device)
    return torch_module.randn(shape, generator=generator, device=device, dtype=dtype)
