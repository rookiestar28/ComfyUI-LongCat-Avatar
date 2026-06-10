from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from typing import Any

from .checkpoint_contract import (
    build_text_encoder_download_manifest,
    download_missing_checkpoint_assets,
)


EXPECTED_TEXT_BATCH = 1
EXPECTED_TEXT_CHANNEL = 1
EXPECTED_TEXT_HIDDEN_SIZE = 4096
MAX_TEXT_SEQUENCE_LENGTH = 512
TEXT_CONDITIONING_SOURCE_CLIP = "comfy_clip_umt5"
TEXT_CONDITIONING_SOURCE_OFFICIAL = "official_longcat_umt5"
DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT = "LongCat-Video"
TEXT_ENCODER_OFFLOAD_DEVICES = ("cpu", "cuda")


@dataclass(frozen=True)
class OfficialTextEncoderLayout:
    root: str
    root_name: str
    tokenizer_dir: str
    text_encoder_dir: str


def normalize_text_encoder_offload_device(offload_device: Any) -> str:
    normalized = str(offload_device or "cpu").lower()
    if normalized not in TEXT_ENCODER_OFFLOAD_DEVICES:
        raise ValueError(
            "text encoder offload_device must be one of "
            + ", ".join(TEXT_ENCODER_OFFLOAD_DEVICES)
            + f"; got {offload_device!r}."
        )
    return normalized


def _safe_relative_model_path(value: str | None) -> str:
    relative = (value or DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT).strip()
    if not relative:
        relative = DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT
    if os.path.isabs(relative):
        raise ValueError("official text_encoder_root must be relative to ComfyUI/models/longcat.")
    normalized = os.path.normpath(relative)
    if normalized == "." or normalized.startswith("..") or os.path.isabs(normalized):
        raise ValueError("official text_encoder_root must stay inside ComfyUI/models/longcat.")
    return normalized


def _root_from_selection(models_longcat_dir: str, selection: str) -> tuple[str, str]:
    candidate = os.path.abspath(os.path.join(models_longcat_dir, selection))
    parts = selection.replace("\\", "/").split("/")
    if "text_encoder" in parts:
        text_index = parts.index("text_encoder")
        root_selection = "/".join(parts[:text_index]) or DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT
        candidate = os.path.abspath(os.path.join(models_longcat_dir, root_selection))
        return candidate, root_selection
    return candidate, selection


def resolve_official_text_encoder_layout(
    text_encoder_root: str | None,
    models_longcat_dir: str,
) -> OfficialTextEncoderLayout:
    selection = _safe_relative_model_path(text_encoder_root)
    models_root = os.path.abspath(models_longcat_dir)
    root, root_name = _root_from_selection(models_root, selection)
    root_real = os.path.realpath(root)
    models_real = os.path.realpath(models_root)
    if root_real != models_real and not root_real.startswith(models_real + os.sep):
        raise ValueError("official text_encoder_root must stay inside ComfyUI/models/longcat.")

    tokenizer_dir = os.path.join(root, "tokenizer")
    text_encoder_dir = os.path.join(root, "text_encoder")
    missing: list[str] = []
    if not os.path.isdir(tokenizer_dir):
        missing.append("tokenizer/")
    if not os.path.isdir(text_encoder_dir):
        missing.append("text_encoder/")
    if missing:
        raise FileNotFoundError(
            "Official UMT5 text encoder assets are missing: "
            + ", ".join(missing)
            + ". Avatar 1.5 reuses the shared base LongCat-Video text encoder; "
            "place the base LongCat-Video checkpoint under ComfyUI/models/longcat/LongCat-Video."
        )

    tokenizer_required = ("tokenizer_config.json",)
    for filename in tokenizer_required:
        if not os.path.isfile(os.path.join(tokenizer_dir, filename)):
            missing.append(f"tokenizer/{filename}")
    if not os.path.isfile(os.path.join(text_encoder_dir, "config.json")):
        missing.append("text_encoder/config.json")
    if not any(
        filename.endswith((".safetensors", ".bin", ".msgpack", ".index.json"))
        for filename in os.listdir(text_encoder_dir)
    ):
        missing.append("text_encoder model weights")
    if missing:
        raise FileNotFoundError(
            "Official UMT5 text encoder assets are incomplete: "
            + ", ".join(missing)
            + ". Use the shared base LongCat-Video text_encoder, not the Avatar-only DiT checkpoint."
        )

    return OfficialTextEncoderLayout(
        root=root,
        root_name=root_name.replace("\\", "/"),
        tokenizer_dir=tokenizer_dir,
        text_encoder_dir=text_encoder_dir,
    )


def resolve_or_download_official_text_encoder_layout(
    text_encoder_root: str | None,
    models_longcat_dir: str,
    *,
    auto_download_missing_text_encoder: bool,
) -> OfficialTextEncoderLayout:
    try:
        return resolve_official_text_encoder_layout(text_encoder_root, models_longcat_dir)
    except FileNotFoundError:
        if not auto_download_missing_text_encoder:
            raise

    selection = _safe_relative_model_path(text_encoder_root)
    root, root_name = _root_from_selection(os.path.abspath(models_longcat_dir), selection)
    if root_name.replace("\\", "/") != DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT:
        raise FileNotFoundError(
            "Automatic official UMT5 text encoder download is only supported for "
            f"{DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT}. Select that root or download custom text assets manually."
        )

    manifest = build_text_encoder_download_manifest(models_longcat_dir)
    # CRITICAL: keep this downloader bounded to the official base LongCat-Video text assets; do not expose arbitrary repos.
    download_missing_checkpoint_assets(manifest)
    return resolve_official_text_encoder_layout(text_encoder_root, models_longcat_dir)


def _shape_tuple(value: Any, role: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{role} must be a tensor-like object with a shape.")
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{role} has an invalid tensor shape.") from exc


def _validate_sequence(seq_len: int, role: str) -> None:
    if seq_len < 1 or seq_len > MAX_TEXT_SEQUENCE_LENGTH:
        raise ValueError(
            f"{role} sequence length must be between 1 and "
            f"{MAX_TEXT_SEQUENCE_LENGTH}; got {seq_len}."
        )


def _validate_hidden(hidden_size: int, role: str) -> None:
    if hidden_size != EXPECTED_TEXT_HIDDEN_SIZE:
        raise ValueError(
            f"{role} hidden size must be {EXPECTED_TEXT_HIDDEN_SIZE}; "
            f"got {hidden_size}."
        )


def _validate_finite(value: Any, role: str) -> None:
    isfinite = getattr(value, "isfinite", None)
    if isfinite is None:
        return
    finite_result = isfinite()
    all_method = getattr(finite_result, "all", None)
    if all_method is not None:
        finite_result = all_method()
    item_method = getattr(finite_result, "item", None)
    if item_method is not None:
        finite_result = item_method()
    if finite_result is False:
        raise ValueError(f"{role} contains NaN or Inf values.")


def validate_scheduled_text_embedding(value: Any, role: str) -> tuple[int, int, int]:
    shape = _shape_tuple(value, role)
    if len(shape) != 3:
        raise ValueError(f"{role} must have rank 3 before reshape; got rank {len(shape)}.")
    batch_size, seq_len, hidden_size = shape
    if batch_size != EXPECTED_TEXT_BATCH:
        raise ValueError(f"{role} batch size must be {EXPECTED_TEXT_BATCH}; got {batch_size}.")
    _validate_sequence(seq_len, role)
    _validate_hidden(hidden_size, role)
    _validate_finite(value, role)
    return shape


def extract_scheduled_text_embedding(scheduled_output: Any, role: str) -> Any:
    if not isinstance(scheduled_output, Sequence) or isinstance(scheduled_output, (str, bytes)):
        raise TypeError(f"{role} scheduled encoder output must be a non-empty sequence.")
    if len(scheduled_output) < 1:
        raise ValueError(f"{role} scheduled encoder output is empty.")
    first = scheduled_output[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)) or len(first) < 1:
        raise ValueError(f"{role} scheduled encoder output is malformed.")
    embedding = first[0]
    validate_scheduled_text_embedding(embedding, role)
    return embedding


def validate_final_text_embedding(value: Any, role: str) -> tuple[int, int, int, int]:
    shape = _shape_tuple(value, role)
    if len(shape) != 4:
        raise ValueError(f"{role} must have rank 4 after reshape; got rank {len(shape)}.")
    batch_size, channel, seq_len, hidden_size = shape
    if batch_size != EXPECTED_TEXT_BATCH:
        raise ValueError(f"{role} batch size must be {EXPECTED_TEXT_BATCH}; got {batch_size}.")
    if channel != EXPECTED_TEXT_CHANNEL:
        raise ValueError(f"{role} channel dimension must be {EXPECTED_TEXT_CHANNEL}; got {channel}.")
    _validate_sequence(seq_len, role)
    _validate_hidden(hidden_size, role)
    _validate_finite(value, role)
    return shape


def validate_text_conditioning_payload(te_cond: Any, *, require_negative: bool = True) -> None:
    if not isinstance(te_cond, Mapping):
        raise TypeError("Text conditioning must be a mapping with prompt embeddings.")

    missing = [
        key
        for key in ("prompt_embeds", "text")
        if key not in te_cond
    ]
    if require_negative and "negative_prompt_embeds" not in te_cond:
        missing.append("negative_prompt_embeds")
    if missing:
        raise KeyError("Text conditioning missing required keys: " + ", ".join(missing))

    prompt_shape = validate_final_text_embedding(te_cond["prompt_embeds"], "prompt_embeds")
    if require_negative:
        negative_shape = validate_final_text_embedding(
            te_cond["negative_prompt_embeds"],
            "negative_prompt_embeds",
        )
        if prompt_shape != negative_shape:
            raise ValueError(
                "prompt_embeds and negative_prompt_embeds must have identical shapes; "
                f"got {prompt_shape} and {negative_shape}."
            )

    text = te_cond["text"]
    if not isinstance(text, Sequence) or isinstance(text, (str, bytes)) or len(text) != 2:
        raise ValueError("Text conditioning text metadata must contain prompt and negative prompt entries.")


def _prompt_clean(text: str) -> str:
    import ftfy
    import html
    import re

    fixed = ftfy.fix_text(text)
    unescaped = html.unescape(html.unescape(fixed))
    return re.sub(r"\s+", " ", unescaped).strip()


def _encode_prompt_with_official_umt5(
    *,
    tokenizer: Any,
    text_encoder: Any,
    prompt: str,
    text_encoder_device: Any,
    output_device: Any,
    dtype: Any,
    max_sequence_length: int = MAX_TEXT_SEQUENCE_LENGTH,
) -> Any:
    text_inputs = tokenizer(
        [_prompt_clean(prompt)],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(text_encoder_device)
    mask = text_inputs.attention_mask.to(text_encoder_device)
    hidden = text_encoder(input_ids, mask).last_hidden_state
    hidden = hidden.to(dtype=dtype, device=output_device)
    _, seq_len, _ = hidden.shape
    return hidden.view(1, 1, seq_len, -1)


def encode_official_text_conditioning(
    *,
    layout: OfficialTextEncoderLayout,
    prompt: str,
    negative_prompt: str,
    device: Any,
    offload_device: Any = "cpu",
    dtype: Any | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoTokenizer, UMT5EncoderModel
    except Exception as exc:  # pragma: no cover - exercised only in real runtime environments.
        raise RuntimeError(
            "Official LongCat UMT5 text encoding requires transformers with UMT5EncoderModel "
            "and a working torch installation."
        ) from exc

    dtype = dtype or torch.bfloat16
    text_encoder_device = torch.device(normalize_text_encoder_offload_device(offload_device))
    output_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(layout.root, subfolder="tokenizer", torch_dtype=dtype)
    text_encoder = UMT5EncoderModel.from_pretrained(layout.root, subfolder="text_encoder", torch_dtype=dtype)
    # CRITICAL: default to CPU for the native UMT5 encoder; the raw text encoder can exceed 20GB VRAM.
    text_encoder = text_encoder.eval().to(text_encoder_device)
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)

    try:
        with torch.no_grad():
            prompt_embeds = _encode_prompt_with_official_umt5(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt=prompt,
                text_encoder_device=text_encoder_device,
                output_device=output_device,
                dtype=dtype,
            )
            negative_prompt_embeds = _encode_prompt_with_official_umt5(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt=negative_prompt,
                text_encoder_device=text_encoder_device,
                output_device=output_device,
                dtype=dtype,
            )
    finally:
        del text_encoder
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    te_cond = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "text": [prompt, negative_prompt],
        "conditioning_source": TEXT_CONDITIONING_SOURCE_OFFICIAL,
        "text_encoder_root": layout.root_name,
        "text_encoder_offload_device": str(text_encoder_device),
    }
    validate_text_conditioning_payload(te_cond, require_negative=True)
    return te_cond
