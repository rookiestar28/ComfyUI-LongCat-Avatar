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

from LongCat_Video.attention_contract import ATTENTION_MODE_SDPA, validate_attention_mode_for_device
from LongCat_Video.backend_capabilities import (
    describe_backend,
    format_memory_fields,
    mps_cpu_fallback_enabled,
    normalize_backend_type,
    read_memory_stats,
    synchronize,
)
from LongCat_Video.backend_dtype_policy import resolve_backend_dtype_policy


STATUS_PASS = "pass"
STATUS_BLOCKED = "blocked"
DEFAULT_DIT_HIDDEN_SHAPE = (1, 16, 2, 2, 2)
DEFAULT_TEXT_TOKENS = 1


@dataclass(frozen=True)
class DitSmokeMemorySnapshot:
    label: str
    available: bool
    fields: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class DitSmokeResult:
    status: str
    stage: str
    native_mps: bool
    cpu_fallback_enabled: bool
    environment: dict[str, str]
    model_source: str
    attention_backend: str
    dtype_policy: dict[str, str]
    boundary_tensors: dict[str, dict[str, Any]]
    output_shape: tuple[int, ...] | None
    output_dtype: str | None
    memory: tuple[DitSmokeMemorySnapshot, ...]
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
            "model_source": self.model_source,
            "attention_backend": self.attention_backend,
            "dtype_policy": dict(self.dtype_policy),
            "boundary_tensors": dict(self.boundary_tensors),
            "output_shape": list(self.output_shape) if self.output_shape is not None else None,
            "output_dtype": self.output_dtype,
            "memory": [asdict(snapshot) for snapshot in self.memory],
            "detail": self.detail,
            "error_type": self.error_type,
            "traceback_location": list(self.traceback_location),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def run_mps_dit_attention_smoke(
    *,
    checkpoint_root: str | os.PathLike[str] | None = None,
    subfolder: str = "base_model_int8",
    model_source: str = "official_sharded_int8",
    attention_mode: str = ATTENTION_MODE_SDPA,
    hidden_shape: tuple[int, int, int, int, int] = DEFAULT_DIT_HIDDEN_SHAPE,
    text_tokens: int = DEFAULT_TEXT_TOKENS,
    device: str = "mps",
    torch_module: Any = None,
    environ: Mapping[str, str] | None = None,
    dit_loader: Callable[..., Any] | None = None,
    dit_model: Any = None,
) -> DitSmokeResult:
    start = time.monotonic()
    environ = environ or os.environ
    fallback_enabled = mps_cpu_fallback_enabled(environ)
    backend = normalize_backend_type(device)
    torch_module, torch_detail = _resolve_torch(torch_module)
    environment = _environment(torch_detail=torch_detail)
    memory: list[DitSmokeMemorySnapshot] = []
    dtype_policy: dict[str, str] = {}
    boundary_tensors: dict[str, dict[str, Any]] = {}
    output_shape: tuple[int, ...] | None = None
    output_dtype: str | None = None

    if backend != "mps":
        return _blocked(
            start=start,
            stage="backend",
            fallback_enabled=fallback_enabled,
            environment=environment,
            model_source=model_source,
            attention_backend=attention_mode,
            dtype_policy=dtype_policy,
            boundary_tensors=boundary_tensors,
            output_shape=output_shape,
            output_dtype=output_dtype,
            memory=memory,
            detail=f"DiT smoke requires MPS; got backend '{backend}'.",
        )
    if torch_module is None:
        return _blocked(
            start=start,
            stage="dependency",
            fallback_enabled=fallback_enabled,
            environment=environment,
            model_source=model_source,
            attention_backend=attention_mode,
            dtype_policy=dtype_policy,
            boundary_tensors=boundary_tensors,
            output_shape=output_shape,
            output_dtype=output_dtype,
            memory=memory,
            detail="PyTorch is not installed.",
        )
    if fallback_enabled:
        return _blocked(
            start=start,
            stage="environment",
            fallback_enabled=fallback_enabled,
            environment=environment,
            model_source=model_source,
            attention_backend=attention_mode,
            dtype_policy=dtype_policy,
            boundary_tensors=boundary_tensors,
            output_shape=output_shape,
            output_dtype=output_dtype,
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
            model_source=model_source,
            attention_backend=attention_mode,
            dtype_policy=dtype_policy,
            boundary_tensors=boundary_tensors,
            output_shape=output_shape,
            output_dtype=output_dtype,
            memory=memory,
            detail="MPS backend is not available.",
        )

    try:
        attention_status = validate_attention_mode_for_device(attention_mode, device)
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
        dit_dtype = policy.dit_dtype
    except Exception as exc:
        return _blocked_from_exception(
            start=start,
            stage="contract",
            fallback_enabled=fallback_enabled,
            environment=environment,
            model_source=model_source,
            attention_backend=attention_mode,
            dtype_policy=dtype_policy,
            boundary_tensors=boundary_tensors,
            output_shape=output_shape,
            output_dtype=output_dtype,
            memory=memory,
            exc=exc,
        )

    try:
        _capture_memory(memory, "before_load", device, torch_module)
        dit = dit_model if dit_model is not None else _load_dit_model(
            checkpoint_root=checkpoint_root,
            subfolder=subfolder,
            attention_mode=attention_mode,
            dit_loader=dit_loader,
        )
        dit = _move_model_to_device(dit.eval(), device=device)
        _capture_memory(memory, "after_load", device, torch_module)

        tensors = _build_boundary_tensors(dit, hidden_shape, text_tokens, device=device, dtype=dit_dtype, torch_module=torch_module)
        boundary_tensors = _tensor_metadata(tensors)
        with _inference_context(torch_module):
            output = dit(
                hidden_states=tensors["latent"],
                timestep=tensors["timestep"],
                encoder_hidden_states=tensors["prompt"],
                encoder_attention_mask=None,
                audio_embs=tensors["audio"],
            )
            synchronize(device, torch_module=torch_module)
        output_shape = _shape_tuple(output)
        output_dtype = _dtype_name(output)
        _capture_memory(memory, "after_forward", device, torch_module)
    except Exception as exc:
        _capture_memory(memory, "after_failure", device, torch_module)
        return _blocked_from_exception(
            start=start,
            stage=_failure_stage(exc),
            fallback_enabled=fallback_enabled,
            environment=environment,
            model_source=model_source,
            attention_backend=attention_status.mode,
            dtype_policy=dtype_policy,
            boundary_tensors=boundary_tensors,
            output_shape=output_shape,
            output_dtype=output_dtype,
            memory=memory,
            exc=exc,
        )

    return DitSmokeResult(
        status=STATUS_PASS,
        stage="complete",
        native_mps=True,
        cpu_fallback_enabled=fallback_enabled,
        environment=environment,
        model_source=model_source,
        attention_backend=attention_status.mode,
        dtype_policy=dtype_policy,
        boundary_tensors=boundary_tensors,
        output_shape=output_shape,
        output_dtype=output_dtype,
        memory=tuple(memory),
        detail="DiT attention/modulation forward boundary completed on MPS with fallback disabled.",
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


def _load_dit_model(
    *,
    checkpoint_root: str | os.PathLike[str] | None,
    subfolder: str,
    attention_mode: str,
    dit_loader: Callable[..., Any] | None,
) -> Any:
    if checkpoint_root is None:
        raise ValueError("checkpoint_root is required when dit_model is not injected.")
    root = Path(checkpoint_root)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint_root not found: {root.name}")
    loader = dit_loader
    if loader is None:
        from LongCat_Video.longcat_video.modules.quantization import load_quantized_dit

        loader = load_quantized_dit
    return loader(str(root), subfolder=subfolder, single_file=None, attention_mode=attention_mode, cp_split_hw=[1, 1])


def _move_model_to_device(model: Any, *, device: str) -> Any:
    try:
        return model.to(device=device)
    except TypeError:
        return model.to(device)


def _build_boundary_tensors(
    dit: Any,
    hidden_shape: tuple[int, int, int, int, int],
    text_tokens: int,
    *,
    device: str,
    dtype: Any,
    torch_module: Any,
) -> dict[str, Any]:
    batch, _, latent_frames, _, _ = hidden_shape
    config = getattr(dit, "config", None)
    caption_channels = int(_config_value(config, dit, "caption_channels", 4096))
    audio_window = int(_config_value(config, dit, "audio_window", 5))
    audio_block = int(_config_value(config, dit, "audio_block", 12))
    audio_channel = int(_config_value(config, dit, "audio_channel", 768))
    vae_scale = int(_config_value(config, dit, "vae_scale", 4))
    audio_frames = 1 + max(latent_frames - 1, 0) * vae_scale

    return {
        "latent": torch_module.zeros(hidden_shape, device=device, dtype=dtype),
        "timestep": torch_module.zeros((batch,), device=device, dtype=dtype),
        "prompt": torch_module.zeros((batch, 1, text_tokens, caption_channels), device=device, dtype=dtype),
        "negative_prompt": torch_module.zeros((batch, 1, text_tokens, caption_channels), device=device, dtype=dtype),
        "audio": torch_module.zeros(
            (batch, audio_frames, audio_window, audio_block, audio_channel),
            device=device,
            dtype=dtype,
        ),
    }


def _tensor_metadata(tensors: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(_shape_tuple(value) or ()),
            "dtype": _dtype_name(value),
            "device": _device_name(value),
        }
        for name, value in tensors.items()
    }


def _config_value(config: Any, owner: Any, name: str, default: Any) -> Any:
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    return getattr(owner, name, default)


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(part) for part in shape)


def _dtype_name(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype).replace("torch.", "")


def _device_name(value: Any) -> str | None:
    device = getattr(value, "device", None)
    return None if device is None else str(device)


def _inference_context(torch_module: Any) -> Any:
    for name in ("inference_mode", "no_grad"):
        factory = getattr(torch_module, name, None)
        if callable(factory):
            return factory()
    return nullcontext()


def _capture_memory(memory: list[DitSmokeMemorySnapshot], label: str, device: str, torch_module: Any) -> None:
    stats = read_memory_stats(device, torch_module=torch_module)
    memory.append(
        DitSmokeMemorySnapshot(
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
    model_source: str,
    attention_backend: str,
    dtype_policy: dict[str, str],
    boundary_tensors: dict[str, dict[str, Any]],
    output_shape: tuple[int, ...] | None,
    output_dtype: str | None,
    memory: list[DitSmokeMemorySnapshot],
    detail: str,
    error_type: str | None = None,
    traceback_location: tuple[str, ...] = (),
) -> DitSmokeResult:
    return DitSmokeResult(
        status=STATUS_BLOCKED,
        stage=stage,
        native_mps=False,
        cpu_fallback_enabled=fallback_enabled,
        environment=environment,
        model_source=model_source,
        attention_backend=attention_backend,
        dtype_policy=dtype_policy,
        boundary_tensors=boundary_tensors,
        output_shape=output_shape,
        output_dtype=output_dtype,
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
    model_source: str,
    attention_backend: str,
    dtype_policy: dict[str, str],
    boundary_tensors: dict[str, dict[str, Any]],
    output_shape: tuple[int, ...] | None,
    output_dtype: str | None,
    memory: list[DitSmokeMemorySnapshot],
    exc: Exception,
) -> DitSmokeResult:
    return _blocked(
        start=start,
        stage=stage,
        fallback_enabled=fallback_enabled,
        environment=environment,
        model_source=model_source,
        attention_backend=attention_backend,
        dtype_policy=dtype_policy,
        boundary_tensors=boundary_tensors,
        output_shape=output_shape,
        output_dtype=output_dtype,
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
    if "load_quantized_dit" in locations or "safe_load" in locations:
        return "load_model"
    if "_build_boundary_tensors" in locations or "zeros" in locations:
        return "input_tensors"
    if "__call__" in locations or "forward" in locations:
        return "forward"
    return "runtime"
