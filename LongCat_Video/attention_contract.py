from __future__ import annotations

from dataclasses import dataclass
import importlib

from LongCat_Video.backend_capabilities import mps_cpu_fallback_enabled, normalize_backend_type


ATTENTION_MODE_AUTO = "auto"
ATTENTION_MODE_SDPA = "sdpa"
ATTENTION_MODE_FLASH_ATTN_2 = "flash_attn_2"
ATTENTION_MODE_FLASH_ATTN_3 = "flash_attn_3"
ATTENTION_MODE_XFORMERS = "xformers"
ATTENTION_MODE_SAGEATTN = "sageattn"
ATTENTION_MODE_SAGEATTN_3 = "sageattn_3"

ATTENTION_MODES = (
    ATTENTION_MODE_AUTO,
    ATTENTION_MODE_SDPA,
    ATTENTION_MODE_FLASH_ATTN_2,
    ATTENTION_MODE_FLASH_ATTN_3,
    ATTENTION_MODE_XFORMERS,
    ATTENTION_MODE_SAGEATTN,
    ATTENTION_MODE_SAGEATTN_3,
)
MPS_SAFE_ATTENTION_MODES = (ATTENTION_MODE_SDPA,)
MPS_VISIBLE_ATTENTION_MODES = MPS_SAFE_ATTENTION_MODES

_ATTENTION_FLAG_DEFAULTS = {
    "enable_flashattn2": False,
    "enable_flashattn3": False,
    "enable_xformers": False,
    "enable_sageattn": False,
    "enable_sageattn3": False,
}

_ATTENTION_MODE_FLAGS = {
    ATTENTION_MODE_FLASH_ATTN_2: "enable_flashattn2",
    ATTENTION_MODE_FLASH_ATTN_3: "enable_flashattn3",
    ATTENTION_MODE_XFORMERS: "enable_xformers",
    ATTENTION_MODE_SAGEATTN: "enable_sageattn",
    ATTENTION_MODE_SAGEATTN_3: "enable_sageattn3",
}

_BACKEND_CANDIDATES = {
    ATTENTION_MODE_FLASH_ATTN_2: (("flash_attn", "flash_attn_func"),),
    ATTENTION_MODE_FLASH_ATTN_3: (("flash_attn_interface", "flash_attn_func"),),
    ATTENTION_MODE_XFORMERS: (("xformers.ops", "memory_efficient_attention"),),
    ATTENTION_MODE_SAGEATTN: (
        ("sageattention", "sageattn"),
        ("sageattn", "sageattn"),
    ),
    ATTENTION_MODE_SAGEATTN_3: (
        ("sageattn3", "sageattn3_blackwell"),
        ("sageattention", "sageattn_blackwell"),
        ("sageattn", "sageattn_blackwell"),
    ),
}


@dataclass(frozen=True)
class AttentionBackendStatus:
    mode: str
    available: bool
    backend: str | None
    reason: str


def normalize_attention_mode(attention_mode):
    if attention_mode is None or attention_mode == "":
        return ATTENTION_MODE_AUTO
    if attention_mode not in ATTENTION_MODES:
        raise ValueError(f"Unsupported attention_mode: {attention_mode}")
    return attention_mode


def apply_attention_mode_to_config(config, attention_mode):
    mode = normalize_attention_mode(attention_mode)
    updated = dict(config)
    if mode == ATTENTION_MODE_AUTO:
        return updated

    updated.update(attention_mode_config_overrides(mode))
    return updated


def attention_mode_config_overrides(attention_mode):
    mode = normalize_attention_mode(attention_mode)
    if mode == ATTENTION_MODE_AUTO:
        return {}

    updated = dict(_ATTENTION_FLAG_DEFAULTS)
    if mode == ATTENTION_MODE_FLASH_ATTN_2:
        updated["enable_flashattn2"] = True
    elif mode == ATTENTION_MODE_FLASH_ATTN_3:
        updated["enable_flashattn3"] = True
    elif mode == ATTENTION_MODE_XFORMERS:
        updated["enable_xformers"] = True
    elif mode == ATTENTION_MODE_SAGEATTN:
        updated["enable_sageattn"] = True
    elif mode == ATTENTION_MODE_SAGEATTN_3:
        updated["enable_sageattn3"] = True
    elif mode != ATTENTION_MODE_SDPA:
        raise ValueError(f"Unsupported attention_mode: {mode}")
    return updated


def _callable_backend(module_name: str, attr_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    candidate = getattr(module, attr_name, None)
    if callable(candidate):
        return f"{module_name}.{attr_name}"
    return None


def inspect_attention_backend(attention_mode) -> AttentionBackendStatus:
    mode = normalize_attention_mode(attention_mode)
    if mode == ATTENTION_MODE_AUTO:
        return AttentionBackendStatus(mode, True, None, "auto preserves checkpoint config")
    if mode == ATTENTION_MODE_SDPA:
        return AttentionBackendStatus(mode, True, "torch.nn.functional.scaled_dot_product_attention", "built in")
    if mode == ATTENTION_MODE_SAGEATTN_3:
        backend = _first_available_backend(_BACKEND_CANDIDATES[mode])
        if backend is None:
            return AttentionBackendStatus(
                mode,
                False,
                None,
                "requires sageattn3.sageattn3_blackwell or sageattention.sageattn_blackwell",
            )
        return AttentionBackendStatus(
            mode,
            False,
            backend,
            "SageAttention3 is not fully wired for LongCat text varlen cross-attention; use sageattn",
        )

    candidates = _BACKEND_CANDIDATES.get(mode)
    if candidates is None:
        raise ValueError(f"Unsupported attention_mode: {mode}")
    backend = _first_available_backend(candidates)
    if backend is None:
        requirements = ", ".join(f"{module}.{attr}" for module, attr in candidates)
        return AttentionBackendStatus(mode, False, None, f"requires one of: {requirements}")
    return AttentionBackendStatus(mode, True, backend, "available")


def _first_available_backend(candidates: tuple[tuple[str, str], ...]) -> str | None:
    for module_name, attr_name in candidates:
        backend = _callable_backend(module_name, attr_name)
        if backend is not None:
            return backend
    return None


def validate_attention_mode_availability(attention_mode) -> AttentionBackendStatus:
    status = inspect_attention_backend(attention_mode)
    if not status.available:
        raise RuntimeError(
            f"attention_mode '{status.mode}' is not available: {status.reason}. "
            "Install the required package in the same Python environment that launches ComfyUI, "
            "or choose 'auto', 'sdpa', or another available backend."
        )
    return status


def validate_attention_mode_for_device(attention_mode, device) -> AttentionBackendStatus:
    mode = normalize_attention_mode(attention_mode)
    if normalize_backend_type(device) == "mps" and mode not in MPS_SAFE_ATTENTION_MODES:
        raise RuntimeError(
            f"attention_mode '{mode}' is not supported on MPS. "
            "MPS attention is limited to explicit 'sdpa' until Apple Silicon CPU-vs-MPS "
            "shape probes pass without CPU fallback."
        )
    return validate_attention_mode_availability(mode)


def attention_config_flags(config) -> dict[str, bool]:
    return {key: bool(config.get(key, False)) for key in _ATTENTION_FLAG_DEFAULTS}


def attention_diagnostic_lines(attention_mode, config, *, device=None, dtype=None) -> tuple[str, ...]:
    mode = normalize_attention_mode(attention_mode)
    flags = attention_config_flags(config)
    enabled_modes = [
        candidate_mode
        for candidate_mode, flag_name in _ATTENTION_MODE_FLAGS.items()
        if flags.get(flag_name, False)
    ]
    status = inspect_attention_backend(mode)
    lines = [
        f"[INFO] LongCat attention_mode requested: {mode}",
        "[INFO] LongCat attention runtime: "
        f"device={normalize_backend_type(device) if device is not None else 'unknown'}, "
        f"dtype={dtype if dtype is not None else 'unknown'}, "
        f"mps_cpu_fallback_enabled={mps_cpu_fallback_enabled()}",
        "[INFO] LongCat attention flags: "
        + ", ".join(f"{key}={value}" for key, value in flags.items()),
        f"[INFO] LongCat attention backend '{status.mode}': "
        f"{'available' if status.available else 'unavailable'}"
        + (f" via {status.backend}" if status.backend else "")
        + f" ({status.reason})",
    ]
    if mode == ATTENTION_MODE_AUTO:
        if enabled_modes:
            lines.append("[INFO] LongCat auto attention config enables: " + ", ".join(enabled_modes))
        else:
            lines.append("[INFO] LongCat auto attention config enables no optional backend; SDPA will be used.")
    return tuple(lines)


def attention_config_fallback_warnings(attention_mode, config) -> tuple[str, ...]:
    mode = normalize_attention_mode(attention_mode)
    if mode != ATTENTION_MODE_AUTO:
        return ()

    warnings = []
    for candidate_mode, flag_name in _ATTENTION_MODE_FLAGS.items():
        if not bool(config.get(flag_name, False)):
            continue
        status = inspect_attention_backend(candidate_mode)
        if not status.available:
            warnings.append(
                f"[WARN] LongCat auto attention requested '{candidate_mode}' from checkpoint config, "
                f"but it is unavailable ({status.reason}); runtime will fall back to SDPA where supported."
            )
    return tuple(warnings)


def print_attention_diagnostics(attention_mode, config) -> None:
    for line in attention_diagnostic_lines(attention_mode, config):
        print(line)
    for line in attention_config_fallback_warnings(attention_mode, config):
        print(line)
