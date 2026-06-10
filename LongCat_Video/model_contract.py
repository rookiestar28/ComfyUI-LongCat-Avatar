from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, Sequence

from .checkpoint_contract import (
    OFFICIAL_INT8_SHARDED,
    OFFICIAL_SHARDED,
    SINGLE_FILE_SAFETENSORS,
    validate_checkpoint_source,
)


AVATAR_V15 = "avatar-v1.5"
AVATAR_V10 = "avatar-v1.0"
SUPPORTED_MODEL_TYPES = (AVATAR_V15,)
OFFICIAL_V15_DISTILL_STEPS = 8
OFFICIAL_V15_DISTILL_TEXT_CFG = 1.0
OFFICIAL_V15_DISTILL_AUDIO_CFG = 1.0
AVATAR_MIN_INFERENCE_STEPS = 1
AVATAR_MAX_INFERENCE_STEPS = 50
AVATAR_MIN_GUIDANCE_SCALE = 1.0
AVATAR_MAX_GUIDANCE_SCALE = 10.0
OFFICIAL_V15_SAVE_FPS = 25
OFFICIAL_V15_AUDIO_STRIDE = 1
SUPPORTED_DIT_EXTENSIONS = (".safetensors",)
UNSUPPORTED_GGUF_EXTENSION = ".gguf"


@dataclass(frozen=True)
class AvatarModelContract:
    model_type: str
    model_format: str
    model_path: str | None
    vae_path: str
    distill_checkpoint_path: str
    metadata_root: str
    tokenizer_subfolder: str
    scheduler_subfolder: str
    scheduler_source: str
    vae_config_path: str
    dit_config_dir: str
    int8_config_dir: str
    use_distill: bool
    use_int8: bool
    effective_num_inference_steps: int
    effective_text_guidance_scale: float
    effective_audio_guidance_scale: float
    save_fps: int
    audio_stride: int
    source_kind: str = SINGLE_FILE_SAFETENSORS
    checkpoint_root: str | None = None
    checkpoint_subfolder: str | None = None
    checkpoint_index_path: str | None = None
    checkpoint_shard_paths: tuple[str, ...] = ()


def safe_display_path(path: str | os.PathLike[str] | None) -> str:
    if path is None:
        return "<not selected>"
    text = os.fspath(path)
    if not text or text == "none":
        return "<not selected>"
    normalized = os.path.normpath(text)
    name = os.path.basename(normalized)
    return name or normalized


def _require_model_type(model_type: str) -> str:
    if model_type == AVATAR_V15:
        return model_type
    if model_type == AVATAR_V10:
        raise ValueError(
            "Avatar v1.0 is not supported by this ComfyUI contract yet; "
            "select Avatar v1.5 model assets."
        )
    raise ValueError(
        f"Unsupported model_type '{model_type}'. Expected '{AVATAR_V15}'."
    )


def _require_file(path: str | None, label: str) -> str:
    if not path:
        raise FileNotFoundError(f"Missing {label}: no file was selected.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {label}: {safe_display_path(path)}")
    return path


def _require_dir(path: str, label: str) -> str:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing {label}: {safe_display_path(path)}")
    return path


def _require_metadata_file(path: str, label: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {label}: {safe_display_path(path)}")
    return path


def detect_dit_format(model_path: str | None) -> str:
    if not model_path:
        raise FileNotFoundError("Missing DiT model: no diffusion model was selected.")
    extension = os.path.splitext(model_path)[1].lower()
    if extension in SUPPORTED_DIT_EXTENSIONS:
        return extension.lstrip(".")
    if extension == UNSUPPORTED_GGUF_EXTENSION:
        raise ValueError(
            "GGUF DiT loading is not supported by this ComfyUI node yet; "
            "select a .safetensors Avatar v1.5 DiT model."
        )
    raise ValueError(
        f"Unsupported DiT model format '{extension or '<none>'}' for "
        f"{safe_display_path(model_path)}. Expected .safetensors."
    )


def normalize_sampling_parameters(
    model_type: str,
    use_distill: bool,
    num_inference_steps: int,
    text_guidance_scale: float,
    audio_guidance_scale: float,
) -> tuple[int, float, float]:
    try:
        num_inference_steps = int(num_inference_steps)
        text_guidance_scale = float(text_guidance_scale)
        audio_guidance_scale = float(audio_guidance_scale)
    except (TypeError, ValueError) as exc:
        raise TypeError("Sampler steps and guidance scales must be numeric.") from exc

    if not AVATAR_MIN_INFERENCE_STEPS <= num_inference_steps <= AVATAR_MAX_INFERENCE_STEPS:
        raise ValueError(
            "Sampler steps must be between "
            f"{AVATAR_MIN_INFERENCE_STEPS} and {AVATAR_MAX_INFERENCE_STEPS}."
        )
    if not AVATAR_MIN_GUIDANCE_SCALE <= text_guidance_scale <= AVATAR_MAX_GUIDANCE_SCALE:
        raise ValueError(
            "text_guidance_scale must be between "
            f"{AVATAR_MIN_GUIDANCE_SCALE} and {AVATAR_MAX_GUIDANCE_SCALE}."
        )
    if not AVATAR_MIN_GUIDANCE_SCALE <= audio_guidance_scale <= AVATAR_MAX_GUIDANCE_SCALE:
        raise ValueError(
            "audio_guidance_scale must be between "
            f"{AVATAR_MIN_GUIDANCE_SCALE} and {AVATAR_MAX_GUIDANCE_SCALE}."
        )

    if model_type == AVATAR_V15 and use_distill:
        # CRITICAL: official Avatar 1.5 DMD/distill scripts force 8 steps and CFG 1.0.
        return (
            OFFICIAL_V15_DISTILL_STEPS,
            OFFICIAL_V15_DISTILL_TEXT_CFG,
            OFFICIAL_V15_DISTILL_AUDIO_CFG,
        )

    return num_inference_steps, text_guidance_scale, audio_guidance_scale


def resolve_avatar_model_contract(
    model_path: str | None,
    vae_path: str | None,
    distill_checkpoint_path: str | None,
    node_longcat_path: str,
    *,
    use_int8: bool = False,
    model_type: str = AVATAR_V15,
    checkpoint_source: str = SINGLE_FILE_SAFETENSORS,
    official_checkpoint_path: str | None = None,
) -> AvatarModelContract:
    if checkpoint_source == OFFICIAL_INT8_SHARDED:
        use_int8 = True
    if checkpoint_source == OFFICIAL_SHARDED and use_int8:
        raise ValueError("Official non-INT8 sharded mode cannot be combined with use_int8.")
    if use_int8 and model_type != AVATAR_V15:
        raise ValueError("INT8 inference is only supported for Avatar v1.5.")

    resolved_model_type = _require_model_type(model_type)
    checkpoint_spec = None
    if checkpoint_source == SINGLE_FILE_SAFETENSORS:
        model_format = detect_dit_format(model_path)
    elif checkpoint_source in (OFFICIAL_SHARDED, OFFICIAL_INT8_SHARDED):
        checkpoint_spec = validate_checkpoint_source(
            checkpoint_source,
            official_checkpoint_path,
            model_type=resolved_model_type,
        )
        model_format = "sharded_safetensors"
    else:
        raise ValueError(f"Unsupported checkpoint source: {checkpoint_source}")

    if resolved_model_type == AVATAR_V15 and not distill_checkpoint_path:
        raise FileNotFoundError(
            "Avatar v1.5 requires a distill LoRA checkpoint "
            "(dmd_lora.safetensors); select it in the LoRA input."
        )

    metadata_root = os.path.join(node_longcat_path, "LongCat_Video", "LongCat-Video")
    tokenizer_dir = os.path.join(metadata_root, "tokenizer")
    scheduler_dir = os.path.join(metadata_root, "scheduler")
    vae_config_path = os.path.join(metadata_root, "vae", "config.json")
    dit_config_dir = os.path.join(metadata_root, "dit")
    int8_config_dir = os.path.join(metadata_root, "base_model_int8")

    resolved_model_path = (
        _require_file(model_path, "DiT model")
        if checkpoint_source == SINGLE_FILE_SAFETENSORS
        else None
    )
    resolved_vae_path = _require_file(vae_path, "VAE model")
    resolved_lora_path = _require_file(distill_checkpoint_path, "Avatar v1.5 distill LoRA")

    _require_dir(metadata_root, "embedded LongCat metadata")
    _require_dir(tokenizer_dir, "tokenizer metadata")
    _require_metadata_file(os.path.join(tokenizer_dir, "tokenizer_config.json"), "tokenizer config")
    _require_metadata_file(os.path.join(tokenizer_dir, "spiece.model"), "tokenizer model")
    _require_dir(scheduler_dir, "scheduler metadata")
    _require_metadata_file(os.path.join(scheduler_dir, "scheduler_config.json"), "scheduler config")
    _require_metadata_file(vae_config_path, "VAE config")
    _require_dir(dit_config_dir, "DiT config")
    _require_metadata_file(os.path.join(dit_config_dir, "config.json"), "DiT config")

    if use_int8 and checkpoint_source == SINGLE_FILE_SAFETENSORS:
        _require_dir(int8_config_dir, "INT8 model metadata")
        _require_metadata_file(os.path.join(int8_config_dir, "config.json"), "INT8 model config")
        _require_metadata_file(
            os.path.join(int8_config_dir, "quantization_config.json"),
            "INT8 quantization config",
        )

    steps, text_cfg, audio_cfg = normalize_sampling_parameters(
        resolved_model_type,
        True,
        OFFICIAL_V15_DISTILL_STEPS,
        OFFICIAL_V15_DISTILL_TEXT_CFG,
        OFFICIAL_V15_DISTILL_AUDIO_CFG,
    )

    return AvatarModelContract(
        model_type=resolved_model_type,
        model_format=model_format,
        model_path=resolved_model_path,
        vae_path=resolved_vae_path,
        distill_checkpoint_path=resolved_lora_path,
        metadata_root=metadata_root,
        tokenizer_subfolder="tokenizer",
        scheduler_subfolder="scheduler",
        scheduler_source="embedded_comfy_avatar_v15_scheduler",
        vae_config_path=vae_config_path,
        dit_config_dir=dit_config_dir,
        int8_config_dir=int8_config_dir,
        use_distill=True,
        use_int8=bool(use_int8),
        effective_num_inference_steps=steps,
        effective_text_guidance_scale=text_cfg,
        effective_audio_guidance_scale=audio_cfg,
        save_fps=OFFICIAL_V15_SAVE_FPS,
        audio_stride=OFFICIAL_V15_AUDIO_STRIDE,
        source_kind=checkpoint_source,
        checkpoint_root=checkpoint_spec.checkpoint_root if checkpoint_spec is not None else None,
        checkpoint_subfolder=checkpoint_spec.subfolder if checkpoint_spec is not None else None,
        checkpoint_index_path=checkpoint_spec.index_path if checkpoint_spec is not None else None,
        checkpoint_shard_paths=checkpoint_spec.shard_paths if checkpoint_spec is not None else (),
    )


def _filter_keys(keys: Iterable[str], allowed_prefixes: Sequence[str]) -> list[str]:
    if not allowed_prefixes:
        return list(keys)
    return [
        key
        for key in keys
        if not any(key.startswith(prefix) for prefix in allowed_prefixes)
    ]


def validate_state_dict_keys(
    label: str,
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    *,
    allow_missing_prefixes: Sequence[str] = (),
    allow_unexpected_prefixes: Sequence[str] = (),
) -> None:
    remaining_missing = _filter_keys(missing_keys, allow_missing_prefixes)
    remaining_unexpected = _filter_keys(unexpected_keys, allow_unexpected_prefixes)
    if not remaining_missing and not remaining_unexpected:
        return

    details: list[str] = []
    if remaining_missing:
        details.append("missing keys: " + ", ".join(remaining_missing[:10]))
    if remaining_unexpected:
        details.append("unexpected keys: " + ", ".join(remaining_unexpected[:10]))
    raise ValueError(f"{label} state dict mismatch; " + "; ".join(details))


def validate_state_dict_result(
    label: str,
    result: object,
    *,
    allow_missing_prefixes: Sequence[str] = (),
    allow_unexpected_prefixes: Sequence[str] = (),
) -> None:
    missing_keys = getattr(result, "missing_keys", None)
    unexpected_keys = getattr(result, "unexpected_keys", None)
    if missing_keys is None and unexpected_keys is None and isinstance(result, (tuple, list)):
        missing_keys = result[0] if len(result) > 0 else ()
        unexpected_keys = result[1] if len(result) > 1 else ()
    validate_state_dict_keys(
        label,
        missing_keys or (),
        unexpected_keys or (),
        allow_missing_prefixes=allow_missing_prefixes,
        allow_unexpected_prefixes=allow_unexpected_prefixes,
    )
