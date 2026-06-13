from __future__ import annotations

import os
import platform
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from LongCat_Video.backend_capabilities import (
    describe_backend,
    format_memory_fields,
    mps_cpu_fallback_enabled,
    normalize_backend_type,
    read_memory_stats,
    synchronize,
)
from LongCat_Video.backend_dtype_policy import resolve_backend_dtype_policy
from LongCat_Video.model_contract import validate_state_dict_result


STATUS_PASS = "pass"
STATUS_BLOCKED = "blocked"
DEFAULT_VAE_SMOKE_SHAPE = (1, 3, 1, 64, 64)


@dataclass(frozen=True)
class VaeSmokeMemorySnapshot:
    label: str
    available: bool
    fields: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class VaeSmokeResult:
    status: str
    stage: str
    native_mps: bool
    cpu_fallback_enabled: bool
    environment: dict[str, str]
    dtype_policy: dict[str, str]
    sample_shape: tuple[int, ...]
    latent_shape: tuple[int, ...] | None
    decoded_shape: tuple[int, ...] | None
    memory: tuple[VaeSmokeMemorySnapshot, ...]
    detail: str
    error_type: str | None = None
    traceback_location: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stage": self.stage,
            "native_mps": self.native_mps,
            "cpu_fallback_enabled": self.cpu_fallback_enabled,
            "environment": dict(self.environment),
            "dtype_policy": dict(self.dtype_policy),
            "sample_shape": list(self.sample_shape),
            "latent_shape": list(self.latent_shape) if self.latent_shape is not None else None,
            "decoded_shape": list(self.decoded_shape) if self.decoded_shape is not None else None,
            "memory": [asdict(snapshot) for snapshot in self.memory],
            "detail": self.detail,
            "error_type": self.error_type,
            "traceback_location": list(self.traceback_location),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def run_mps_vae_encode_decode_smoke(
    *,
    vae_config_path: str | os.PathLike[str],
    vae_weights_path: str | os.PathLike[str],
    sample_shape: tuple[int, int, int, int, int] = DEFAULT_VAE_SMOKE_SHAPE,
    device: str = "mps",
    torch_module: Any = None,
    environ: Mapping[str, str] | None = None,
    safe_load_fn: Callable[..., Mapping[str, Any]] | None = None,
    autoencoder_cls: Any = None,
) -> VaeSmokeResult:
    start = time.monotonic()
    environ = environ or os.environ
    memory: list[VaeSmokeMemorySnapshot] = []
    fallback_enabled = mps_cpu_fallback_enabled(environ)
    backend = normalize_backend_type(device)
    torch_module, torch_detail = _resolve_torch(torch_module)
    environment = _environment(torch_detail=torch_detail)
    dtype_policy: dict[str, str] = {}
    latent_shape: tuple[int, ...] | None = None
    decoded_shape: tuple[int, ...] | None = None

    if backend != "mps":
        return _blocked(
            start=start,
            stage="backend",
            fallback_enabled=fallback_enabled,
            environment=environment,
            dtype_policy=dtype_policy,
            sample_shape=sample_shape,
            latent_shape=latent_shape,
            decoded_shape=decoded_shape,
            memory=memory,
            detail=f"VAE smoke requires MPS; got backend '{backend}'.",
        )
    if torch_module is None:
        return _blocked(
            start=start,
            stage="dependency",
            fallback_enabled=fallback_enabled,
            environment=environment,
            dtype_policy=dtype_policy,
            sample_shape=sample_shape,
            latent_shape=latent_shape,
            decoded_shape=decoded_shape,
            memory=memory,
            detail="PyTorch is not installed.",
        )
    if fallback_enabled:
        return _blocked(
            start=start,
            stage="environment",
            fallback_enabled=fallback_enabled,
            environment=environment,
            dtype_policy=dtype_policy,
            sample_shape=sample_shape,
            latent_shape=latent_shape,
            decoded_shape=decoded_shape,
            memory=memory,
            detail="PYTORCH_ENABLE_MPS_FALLBACK is enabled; native MPS evidence requires fallback disabled.",
        )

    caps = describe_backend(device, torch_module=torch_module, environ=environ)
    if not caps.available:
        return _blocked(
            start=start,
            stage="backend",
            fallback_enabled=fallback_enabled,
            environment=environment,
            dtype_policy=dtype_policy,
            sample_shape=sample_shape,
            latent_shape=latent_shape,
            decoded_shape=decoded_shape,
            memory=memory,
            detail="MPS backend is not available.",
        )

    try:
        policy = resolve_backend_dtype_policy(device, torch_module=torch_module)
        dtype_policy = {
            "backend": policy.backend,
            "requested": policy.requested_precision,
            "text": policy.text_encoder_precision,
            "audio": policy.audio_encoder_precision,
            "dit": policy.dit_precision,
            "vae": policy.vae_precision,
            "math": policy.math_precision,
            "bf16_probe": policy.bfloat16_probe.detail if policy.bfloat16_probe else "",
        }
        vae_dtype = policy.vae_dtype
    except Exception as exc:
        return _blocked_from_exception(
            start=start,
            stage="dtype_policy",
            fallback_enabled=fallback_enabled,
            environment=environment,
            dtype_policy=dtype_policy,
            sample_shape=sample_shape,
            latent_shape=latent_shape,
            decoded_shape=decoded_shape,
            memory=memory,
            exc=exc,
        )

    try:
        config_path = _require_file("vae_config_path", vae_config_path)
        weights_path = _require_file("vae_weights_path", vae_weights_path)
        autoencoder_cls = _resolve_autoencoder(autoencoder_cls)
        safe_load_fn = _resolve_safe_load(safe_load_fn)

        _capture_memory(memory, "before_load", device, torch_module)
        vae_config = autoencoder_cls.load_config(str(config_path))
        vae = autoencoder_cls.from_config(vae_config, torch_dtype=vae_dtype)
        state_dict = safe_load_fn(str(weights_path), device="cpu")
        load_result = vae.load_state_dict(state_dict, strict=False)
        validate_state_dict_result("VAE", load_result)
        del state_dict

        vae = _move_module_to_device(vae.eval(), device=device, dtype=vae_dtype)
        _capture_memory(memory, "after_load", device, torch_module)

        sample = torch_module.zeros(sample_shape, device=device, dtype=vae_dtype)
        _capture_memory(memory, "after_sample_alloc", device, torch_module)
        with _inference_context(torch_module):
            encoded = vae.encode(sample)
            latent = _extract_latent(encoded)
            latent_shape = _shape_tuple(latent)
            _capture_memory(memory, "after_encode", device, torch_module)
            decoded = vae.decode(latent, return_dict=True)
            decoded_tensor = getattr(decoded, "sample", decoded[0] if isinstance(decoded, (tuple, list)) else decoded)
            decoded_shape = _shape_tuple(decoded_tensor)
            synchronize(device, torch_module=torch_module)
            _capture_memory(memory, "after_decode", device, torch_module)
    except Exception as exc:
        _capture_memory(memory, "after_failure", device, torch_module)
        return _blocked_from_exception(
            start=start,
            stage=_failure_stage(exc),
            fallback_enabled=fallback_enabled,
            environment=environment,
            dtype_policy=dtype_policy,
            sample_shape=sample_shape,
            latent_shape=latent_shape,
            decoded_shape=decoded_shape,
            memory=memory,
            exc=exc,
        )

    return VaeSmokeResult(
        status=STATUS_PASS,
        stage="complete",
        native_mps=True,
        cpu_fallback_enabled=fallback_enabled,
        environment=environment,
        dtype_policy=dtype_policy,
        sample_shape=tuple(sample_shape),
        latent_shape=latent_shape,
        decoded_shape=decoded_shape,
        memory=tuple(memory),
        detail="VAE encode and decode completed on MPS with fallback disabled.",
        elapsed_seconds=time.monotonic() - start,
    )


def _resolve_torch(torch_module: Any) -> tuple[Any, str]:
    if torch_module is not None:
        return torch_module, str(getattr(torch_module, "__version__", "injected"))
    try:
        import torch as imported_torch
    except ModuleNotFoundError:
        return None, "unavailable"
    return imported_torch, str(getattr(imported_torch, "__version__", "unknown"))


def _resolve_safe_load(safe_load_fn: Callable[..., Mapping[str, Any]] | None) -> Callable[..., Mapping[str, Any]]:
    if safe_load_fn is not None:
        return safe_load_fn
    try:
        from safetensors.torch import load_file as safe_load
    except ModuleNotFoundError as exc:
        raise RuntimeError("safetensors is required to load VAE weights for the MPS smoke.") from exc
    return safe_load


def _resolve_autoencoder(autoencoder_cls: Any) -> Any:
    if autoencoder_cls is not None:
        return autoencoder_cls
    try:
        from LongCat_Video.longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    except ModuleNotFoundError as exc:
        raise RuntimeError("AutoencoderKLWan dependencies are unavailable for the MPS VAE smoke.") from exc
    return AutoencoderKLWan


def _require_file(label: str, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path.name}")
    return path


def _move_module_to_device(module: Any, *, device: str, dtype: Any) -> Any:
    try:
        return module.to(device=device, dtype=dtype)
    except TypeError:
        module = module.to(device)
        return module.to(dtype)


def _extract_latent(encoded: Any) -> Any:
    if hasattr(encoded, "latent_dist"):
        latent_dist = encoded.latent_dist
        if hasattr(latent_dist, "mode"):
            return latent_dist.mode()
        if hasattr(latent_dist, "mean"):
            return latent_dist.mean
    if hasattr(encoded, "latents"):
        return encoded.latents
    if isinstance(encoded, (tuple, list)) and encoded:
        return _extract_latent(encoded[0])
    raise AttributeError("Could not access VAE smoke latents from encoder output.")


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(part) for part in shape)


def _inference_context(torch_module: Any) -> Any:
    for name in ("inference_mode", "no_grad"):
        factory = getattr(torch_module, name, None)
        if callable(factory):
            return factory()
    return nullcontext()


def _capture_memory(memory: list[VaeSmokeMemorySnapshot], label: str, device: str, torch_module: Any) -> None:
    stats = read_memory_stats(device, torch_module=torch_module)
    memory.append(
        VaeSmokeMemorySnapshot(
            label=label,
            available=stats.available,
            fields=tuple(format_memory_fields(stats)),
            detail=stats.detail,
        )
    )


def _environment(*, torch_detail: str) -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch_detail,
    }


def _blocked(
    *,
    start: float,
    stage: str,
    fallback_enabled: bool,
    environment: dict[str, str],
    dtype_policy: dict[str, str],
    sample_shape: tuple[int, ...],
    latent_shape: tuple[int, ...] | None,
    decoded_shape: tuple[int, ...] | None,
    memory: list[VaeSmokeMemorySnapshot],
    detail: str,
    error_type: str | None = None,
    traceback_location: tuple[str, ...] = (),
) -> VaeSmokeResult:
    return VaeSmokeResult(
        status=STATUS_BLOCKED,
        stage=stage,
        native_mps=False,
        cpu_fallback_enabled=fallback_enabled,
        environment=environment,
        dtype_policy=dtype_policy,
        sample_shape=tuple(sample_shape),
        latent_shape=latent_shape,
        decoded_shape=decoded_shape,
        memory=tuple(memory),
        detail=detail,
        error_type=error_type,
        traceback_location=traceback_location,
        elapsed_seconds=time.monotonic() - start,
    )


def _blocked_from_exception(
    *,
    start: float,
    stage: str,
    fallback_enabled: bool,
    environment: dict[str, str],
    dtype_policy: dict[str, str],
    sample_shape: tuple[int, ...],
    latent_shape: tuple[int, ...] | None,
    decoded_shape: tuple[int, ...] | None,
    memory: list[VaeSmokeMemorySnapshot],
    exc: Exception,
) -> VaeSmokeResult:
    return _blocked(
        start=start,
        stage=stage,
        fallback_enabled=fallback_enabled,
        environment=environment,
        dtype_policy=dtype_policy,
        sample_shape=sample_shape,
        latent_shape=latent_shape,
        decoded_shape=decoded_shape,
        memory=memory,
        detail=f"{type(exc).__name__}: {exc}",
        error_type=type(exc).__name__,
        traceback_location=_traceback_location(exc),
    )


def _traceback_location(exc: Exception) -> tuple[str, ...]:
    frames = traceback.extract_tb(exc.__traceback__)
    return tuple(f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}" for frame in frames[-5:])


def _failure_stage(exc: Exception) -> str:
    locations = " ".join(_traceback_location(exc))
    if ".encode" in locations or "encode" in locations:
        return "encode"
    if ".decode" in locations or "decode" in locations:
        return "decode"
    if "load_state_dict" in locations or "safe_load" in locations:
        return "load_weights"
    if "from_config" in locations or "load_config" in locations:
        return "load_config"
    return "runtime"
