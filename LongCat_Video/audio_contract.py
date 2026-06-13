from __future__ import annotations

from collections.abc import Mapping
import math
import os
from typing import Any

from LongCat_Video.backend_capabilities import normalize_backend_type


AVATAR_SAMPLE_RATE = 16000
AVATAR_SAVE_FPS = 25
AVATAR_AUDIO_STRIDE = 1
AVATAR_NUM_FRAMES = 93
AVATAR_NUM_COND_FRAMES = 13
AVATAR_AUDIO_LAYERS = 5
AVATAR_AUDIO_HIDDEN_WIDTH = 1280
AVATAR_AUDIO_PAYLOAD_TYPE = "longcat_avatar_audio_full"
MAX_ADVANCED_SPEAKER_TRACKS = 4
AUDIO_TYPE_PARA = "para"
AUDIO_TYPE_ADD = "add"
SUPPORTED_AUDIO_TYPES = (AUDIO_TYPE_PARA, AUDIO_TYPE_ADD)
LONGCAT_AVATAR_WHISPER_ENCODER = "whisper-large-v3.safetensors"
SUPPORTED_LONGCAT_AVATAR_WHISPER_ENCODERS = (LONGCAT_AVATAR_WHISPER_ENCODER,)
_SUPPORTED_LONGCAT_AVATAR_WHISPER_ENCODERS_LOWER = tuple(
    encoder.lower() for encoder in SUPPORTED_LONGCAT_AVATAR_WHISPER_ENCODERS
)
LONGCAT_AVATAR_WHISPER_STATE_DICT_PREFIX = "model."


def resolve_audio_embedding_device(runtime_device: Any) -> Any:
    backend = normalize_backend_type(runtime_device)
    if backend in {"cuda", "mps"}:
        return runtime_device
    return "cpu"


def validate_longcat_avatar_whisper_model_name(audio_encoder_name: str) -> str:
    try:
        selected_name = os.path.basename(os.fspath(audio_encoder_name)).strip()
    except TypeError as exc:
        raise TypeError("LongCat Avatar Whisper audio_encoder must be a model filename.") from exc

    if selected_name.lower() not in _SUPPORTED_LONGCAT_AVATAR_WHISPER_ENCODERS_LOWER:
        raise ValueError(
            "LongCat Avatar Whisper only supports "
            f"{LONGCAT_AVATAR_WHISPER_ENCODER}; selected '{selected_name or '<empty>'}' is not supported. "
            "Wav2Vec2 audio encoders are not compatible with the Avatar 1.5 Whisper node."
        )
    return selected_name


def normalize_longcat_avatar_whisper_state_dict(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(state_dict, Mapping):
        raise TypeError("LongCat Avatar Whisper state dict must be a mapping.")

    keys = tuple(state_dict.keys())
    if not keys:
        raise ValueError("LongCat Avatar Whisper state dict is empty.")
    if not all(isinstance(key, str) for key in keys):
        raise TypeError("LongCat Avatar Whisper state dict keys must be strings.")

    prefixed_keys = [key for key in keys if key.startswith(LONGCAT_AVATAR_WHISPER_STATE_DICT_PREFIX)]
    if not prefixed_keys:
        return state_dict
    if len(prefixed_keys) != len(keys):
        raise ValueError(
            "LongCat Avatar Whisper state dict mixes prefixed and unprefixed keys; "
            "cannot safely normalize checkpoint keys."
        )

    normalized: dict[str, Any] = {}
    prefix_length = len(LONGCAT_AVATAR_WHISPER_STATE_DICT_PREFIX)
    for key, value in state_dict.items():
        normalized_key = key[prefix_length:]
        if not normalized_key:
            raise ValueError("LongCat Avatar Whisper state dict contains an empty normalized key.")
        if normalized_key in normalized:
            raise ValueError(
                "LongCat Avatar Whisper state dict normalization produced duplicate key "
                f"'{normalized_key}'."
            )
        normalized[normalized_key] = value
    return normalized


def validate_longcat_avatar_whisper_load_result(result: Any) -> None:
    missing_keys = getattr(result, "missing_keys", None)
    unexpected_keys = getattr(result, "unexpected_keys", None)
    if missing_keys is None and unexpected_keys is None and isinstance(result, (tuple, list)):
        missing_keys = result[0] if len(result) > 0 else ()
        unexpected_keys = result[1] if len(result) > 1 else ()

    missing = list(missing_keys or ())
    unexpected = list(unexpected_keys or ())
    if not missing and not unexpected:
        return

    details: list[str] = []
    if missing:
        details.append("missing keys: " + ", ".join(missing[:10]))
    if unexpected:
        details.append("unexpected keys: " + ", ".join(unexpected[:10]))
    raise ValueError("LongCat Avatar Whisper state dict mismatch; " + "; ".join(details))


def calculate_generate_duration(
    save_fps: int,
    num_segments: int,
    *,
    num_frames: int = AVATAR_NUM_FRAMES,
    num_cond_frames: int = AVATAR_NUM_COND_FRAMES,
) -> float:
    if save_fps <= 0:
        raise ValueError("save_fps must be greater than 0.")
    if num_segments < 1:
        raise ValueError("num_segments must be at least 1.")
    return num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps


def calculate_num_segments_for_audio_duration(
    audio_duration: float,
    save_fps: int,
    *,
    num_frames: int = AVATAR_NUM_FRAMES,
    num_cond_frames: int = AVATAR_NUM_COND_FRAMES,
) -> int:
    try:
        audio_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise TypeError("audio_duration must be numeric.") from exc
    if audio_duration <= 0:
        raise ValueError("audio_duration must be greater than 0.")
    if save_fps <= 0:
        raise ValueError("save_fps must be greater than 0.")
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than 0.")
    continuation_frames = num_frames - num_cond_frames
    if continuation_frames <= 0:
        raise ValueError("num_cond_frames must be smaller than num_frames.")

    first_duration = num_frames / save_fps
    if audio_duration <= first_duration:
        return 1
    continuation_duration = continuation_frames / save_fps
    return 1 + int(math.ceil((audio_duration - first_duration) / continuation_duration))


def calculate_num_segments_for_sample_count(
    sample_count: int,
    sample_rate: int,
    save_fps: int,
    *,
    num_frames: int = AVATAR_NUM_FRAMES,
    num_cond_frames: int = AVATAR_NUM_COND_FRAMES,
) -> int:
    try:
        sample_count = int(sample_count)
        sample_rate = int(sample_rate)
    except (TypeError, ValueError) as exc:
        raise TypeError("sample_count and sample_rate must be integers.") from exc
    if sample_count <= 0:
        raise ValueError("sample_count must be greater than 0.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0.")
    return calculate_num_segments_for_audio_duration(
        sample_count / sample_rate,
        save_fps,
        num_frames=num_frames,
        num_cond_frames=num_cond_frames,
    )


def calculate_target_output_frames_for_sample_count(sample_count: int, sample_rate: int, save_fps: int) -> int:
    try:
        sample_count = int(sample_count)
        sample_rate = int(sample_rate)
        save_fps = int(save_fps)
    except (TypeError, ValueError) as exc:
        raise TypeError("sample_count, sample_rate, and save_fps must be integers.") from exc
    if sample_count <= 0:
        raise ValueError("sample_count must be greater than 0.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0.")
    if save_fps <= 0:
        raise ValueError("save_fps must be greater than 0.")
    return int(math.ceil(sample_count * save_fps / sample_rate))


def target_sample_count(generate_duration: float, sample_rate: int = AVATAR_SAMPLE_RATE) -> int:
    if generate_duration <= 0:
        raise ValueError("generate_duration must be greater than 0.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0.")
    return int(math.ceil(generate_duration * sample_rate))


def validate_audio_type(audio_type: str) -> str:
    if audio_type not in SUPPORTED_AUDIO_TYPES:
        raise ValueError(f"Unsupported audio_type '{audio_type}'. Expected one of {SUPPORTED_AUDIO_TYPES}.")
    return audio_type


def normalize_optional_lengths(left_len: int | None, right_len: int | None) -> tuple[int, int]:
    if left_len is None and right_len is None:
        raise ValueError("At least one audio clip is required.")
    if left_len is None:
        left_len = right_len
    if right_len is None:
        right_len = left_len
    if left_len is None or right_len is None:
        raise ValueError("At least one audio clip is required.")
    if left_len < 0 or right_len < 0:
        raise ValueError("Audio lengths must be non-negative.")
    return left_len, right_len


def validate_multi_audio_lengths(left_len: int, right_len: int, audio_type: str) -> None:
    validate_audio_type(audio_type)
    if audio_type == AUDIO_TYPE_PARA and left_len != right_len:
        raise ValueError("audio_type='para' requires equal-length left and right audio.")


def calculate_source_sample_count_for_prepared_audio_lengths(
    right_sample_count: int,
    *,
    audio_type: str = AUDIO_TYPE_PARA,
    left_sample_count: int | None = None,
) -> int:
    validate_audio_type(audio_type)
    right_sample_count = int(right_sample_count)
    if right_sample_count <= 0:
        raise ValueError("right_sample_count must be greater than 0.")
    if left_sample_count is None:
        return right_sample_count

    left_sample_count = int(left_sample_count)
    if left_sample_count <= 0:
        raise ValueError("left_sample_count must be greater than 0.")
    if audio_type == AUDIO_TYPE_ADD:
        return left_sample_count + right_sample_count
    validate_multi_audio_lengths(left_sample_count, right_sample_count, audio_type)
    return right_sample_count


def calculate_num_segments_for_prepared_audio_lengths(
    right_sample_count: int,
    sample_rate: int,
    save_fps: int,
    *,
    audio_type: str = AUDIO_TYPE_PARA,
    left_sample_count: int | None = None,
    num_frames: int = AVATAR_NUM_FRAMES,
    num_cond_frames: int = AVATAR_NUM_COND_FRAMES,
) -> int:
    source_sample_count = calculate_source_sample_count_for_prepared_audio_lengths(
        right_sample_count,
        audio_type=audio_type,
        left_sample_count=left_sample_count,
    )
    return calculate_num_segments_for_sample_count(
        source_sample_count,
        sample_rate,
        save_fps,
        num_frames=num_frames,
        num_cond_frames=num_cond_frames,
    )


def prepared_multi_audio_length(
    left_len: int | None,
    right_len: int | None,
    generate_duration: float,
    *,
    sample_rate: int = AVATAR_SAMPLE_RATE,
    audio_type: str = AUDIO_TYPE_PARA,
) -> int:
    left_len, right_len = normalize_optional_lengths(left_len, right_len)
    validate_multi_audio_lengths(left_len, right_len, audio_type)
    if audio_type == AUDIO_TYPE_ADD:
        prepared_len = left_len + right_len
    else:
        prepared_len = left_len
    return max(prepared_len, target_sample_count(generate_duration, sample_rate))


def ensure_mono_waveform_array(waveform: Any, role: str = "audio waveform") -> Any:
    ndim = getattr(waveform, "ndim", None)
    if ndim is None:
        return waveform
    if ndim == 1:
        return waveform
    if ndim == 2:
        mean = getattr(waveform, "mean", None)
        if mean is None:
            raise TypeError(f"{role} must support channel averaging for mono conversion.")
        return mean(axis=0)
    raise ValueError(f"{role} must be mono or channel-first audio; got rank {ndim}.")


def _shape_tuple(value: Any, role: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{role} must be a tensor-like object with a shape.")
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{role} has an invalid shape.") from exc


def _as_bool(value: Any) -> bool:
    item = getattr(value, "item", None)
    if item is not None:
        value = item()
    return bool(value)


def _all(value: Any) -> Any:
    all_method = getattr(value, "all", None)
    return all_method() if all_method is not None else value


def _any(value: Any) -> Any:
    any_method = getattr(value, "any", None)
    return any_method() if any_method is not None else value


def validate_finite_embedding(value: Any, role: str) -> None:
    isfinite = getattr(value, "isfinite", None)
    if isfinite is not None:
        if not _as_bool(_all(isfinite())):
            raise ValueError(f"{role} contains NaN or Inf values.")
        return

    isnan = getattr(value, "isnan", None)
    if isnan is not None and _as_bool(_any(isnan())):
        raise ValueError(f"{role} contains NaN values.")

    isinf = getattr(value, "isinf", None)
    if isinf is not None and _as_bool(_any(isinf())):
        raise ValueError(f"{role} contains Inf values.")


def validate_audio_embedding(value: Any, role: str) -> tuple[int, int, int]:
    shape = _shape_tuple(value, role)
    if len(shape) != 3:
        raise ValueError(f"{role} must have rank 3 [T, 5, D]; got rank {len(shape)}.")
    frames, layers, width = shape
    if frames < 1:
        raise ValueError(f"{role} must contain at least one frame.")
    if layers != AVATAR_AUDIO_LAYERS:
        raise ValueError(f"{role} layer dimension must be {AVATAR_AUDIO_LAYERS}; got {layers}.")
    if width != AVATAR_AUDIO_HIDDEN_WIDTH:
        raise ValueError(f"{role} hidden width must be {AVATAR_AUDIO_HIDDEN_WIDTH}; got {width}.")
    validate_finite_embedding(value, role)
    return shape


def validate_matching_audio_embedding_shapes(*items: tuple[str, Any]) -> tuple[int, int, int]:
    if not items:
        raise ValueError("At least one audio embedding is required.")
    first_role, first_value = items[0]
    expected_shape = validate_audio_embedding(first_value, first_role)
    for role, value in items[1:]:
        shape = validate_audio_embedding(value, role)
        if shape != expected_shape:
            raise ValueError(f"{role} shape must match {first_role}; got {shape} and {expected_shape}.")
    return expected_shape


def _validate_non_negative_number(value: Any, role: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{role} must be numeric.") from exc
    if normalized < 0:
        raise ValueError(f"{role} must be non-negative.")
    return normalized


def validate_avatar_audio_payload_metadata(payload: Mapping[str, Any]) -> None:
    payload_type = payload.get("payload_type")
    if payload_type != AVATAR_AUDIO_PAYLOAD_TYPE:
        raise ValueError(f"Unsupported audio payload_type '{payload_type}'.")

    audio_features = payload.get("audio_features")
    if not isinstance(audio_features, tuple) or not audio_features:
        raise ValueError("Audio payload audio_features must be a non-empty tuple.")

    speaker_roles = payload.get("speaker_roles")
    if not isinstance(speaker_roles, tuple) or len(speaker_roles) != len(audio_features):
        raise ValueError("Audio payload speaker_roles must match audio_features.")
    for role in speaker_roles:
        if not isinstance(role, str) or not role:
            raise ValueError("Audio payload speaker_roles must contain non-empty strings.")

    named_features = tuple(
        (f"audio_features[{idx}:{role}]", emb)
        for idx, (role, emb) in enumerate(zip(speaker_roles, audio_features))
    )
    validate_matching_audio_embedding_shapes(*named_features)

    if int(payload.get("audio_stride", 0)) != AVATAR_AUDIO_STRIDE:
        raise ValueError(f"Avatar v1.5 audio_stride must be {AVATAR_AUDIO_STRIDE}.")
    if int(payload.get("save_fps", 0)) <= 0:
        raise ValueError("Audio payload save_fps must be greater than 0.")
    if int(payload.get("num_segments", 0)) < 1:
        raise ValueError("Audio payload num_segments must be at least 1.")
    if float(payload.get("generate_duration", 0.0)) <= 0:
        raise ValueError("Audio payload generate_duration must be greater than 0.")
    target_output_frames = payload.get("target_output_frames")
    if target_output_frames is not None:
        target_output_frames = int(target_output_frames)
        expected_frames = AVATAR_NUM_FRAMES + (int(payload["num_segments"]) - 1) * (
            AVATAR_NUM_FRAMES - AVATAR_NUM_COND_FRAMES
        )
        if target_output_frames < 1:
            raise ValueError("Audio payload target_output_frames must be at least 1.")
        if target_output_frames > expected_frames:
            raise ValueError("Audio payload target_output_frames exceeds generated segment frame coverage.")

    validate_audio_type(str(payload.get("audio_type", "")))
    _validate_non_negative_number(payload.get("audio_scale", 1.0), "audio_scale")
    _validate_non_negative_number(payload.get("audio_cfg_scale", 1.0), "audio_cfg_scale")

    masks = payload.get("masks")
    if masks is not None and not isinstance(masks, Mapping):
        raise TypeError("Audio payload masks must be a mapping when provided.")
    boxes = payload.get("boxes")
    if boxes is not None and not isinstance(boxes, Mapping):
        raise TypeError("Audio payload boxes must be a mapping when provided.")


def build_avatar_audio_payload(
    *,
    full_audio_emb: Any,
    num_segments: int,
    audio_stride: int = AVATAR_AUDIO_STRIDE,
    save_fps: int = AVATAR_SAVE_FPS,
    audio_type: str = AUDIO_TYPE_PARA,
    left_full_audio_emb: Any | None = None,
    back_full_audio_emb: Any | None = None,
    use_background_silent_audio: bool = False,
    left_person_bbox: Any | None = None,
    right_person_bbox: Any | None = None,
    other_person_bbox: Any | None = None,
    masks: Mapping[str, Any] | None = None,
    audio_scale: float = 1.0,
    audio_cfg_scale: float = 1.0,
    audio_encoder_type: str = "whisper",
    target_output_frames: int | None = None,
) -> dict[str, Any]:
    if int(audio_stride) != AVATAR_AUDIO_STRIDE:
        raise ValueError(f"Avatar v1.5 audio_stride must be {AVATAR_AUDIO_STRIDE}.")
    num_segments = int(num_segments)
    save_fps = int(save_fps)
    generate_duration = calculate_generate_duration(save_fps, num_segments)
    validate_audio_type(audio_type)

    if left_full_audio_emb is None:
        speaker_roles = ("primary",)
        audio_features = (full_audio_emb,)
        validate_audio_embedding(full_audio_emb, "full_audio_emb")
    else:
        speaker_roles = ("left", "right")
        audio_features = (left_full_audio_emb, full_audio_emb)
        embeddings = [
            ("left_full_audio_emb", left_full_audio_emb),
            ("full_audio_emb", full_audio_emb),
        ]
        if use_background_silent_audio:
            if back_full_audio_emb is None:
                raise ValueError("Background silent audio is enabled but back_full_audio_emb is missing.")
            speaker_roles = speaker_roles + ("background",)
            audio_features = audio_features + (back_full_audio_emb,)
            embeddings.append(("back_full_audio_emb", back_full_audio_emb))
        validate_matching_audio_embedding_shapes(*embeddings)

    boxes = {
        key: value
        for key, value in {
            "left": left_person_bbox,
            "right": right_person_bbox,
            "other": other_person_bbox,
        }.items()
        if value is not None
    }

    payload = {
        "payload_type": AVATAR_AUDIO_PAYLOAD_TYPE,
        "audio_features": audio_features,
        "speaker_roles": speaker_roles,
        "audio_encoder_type": audio_encoder_type,
        "audio_scale": float(audio_scale),
        "audio_cfg_scale": float(audio_cfg_scale),
        "audio_type": audio_type,
        "save_fps": save_fps,
        "generate_duration": generate_duration,
        "target_output_frames": None if target_output_frames is None else int(target_output_frames),
        "full_audio_emb": full_audio_emb,
        "num_segments": num_segments,
        "audio_stride": int(audio_stride),
        "back_full_audio_emb": back_full_audio_emb,
        "left_full_audio_emb": left_full_audio_emb,
        "left_person_bbox": left_person_bbox,
        "right_person_bbox": right_person_bbox,
        "other_person_bbox": other_person_bbox,
        "use_background_silent_audio": bool(use_background_silent_audio),
        "masks": masks,
        "boxes": boxes,
    }
    validate_audio_conditioning_payload(payload)
    return payload


def build_multi_speaker_audio_payload(
    speaker_embeddings: Mapping[str, Any],
    *,
    num_segments: int,
    audio_stride: int = AVATAR_AUDIO_STRIDE,
    save_fps: int = AVATAR_SAVE_FPS,
    audio_type: str = AUDIO_TYPE_PARA,
    masks: Mapping[str, Any] | None = None,
    boxes: Mapping[str, Any] | None = None,
    audio_scale: float = 1.0,
    audio_cfg_scale: float = 1.0,
    audio_encoder_type: str = "whisper",
    target_output_frames: int | None = None,
) -> dict[str, Any]:
    if not isinstance(speaker_embeddings, Mapping) or not speaker_embeddings:
        raise ValueError("speaker_embeddings must contain at least one named speaker track.")
    if len(speaker_embeddings) > MAX_ADVANCED_SPEAKER_TRACKS:
        raise ValueError(f"Avatar audio payload supports at most {MAX_ADVANCED_SPEAKER_TRACKS} speaker tracks.")
    if masks is not None and not isinstance(masks, Mapping):
        raise TypeError("masks must be a mapping when provided.")
    if boxes is not None and not isinstance(boxes, Mapping):
        raise TypeError("boxes must be a mapping when provided.")

    speaker_roles = tuple(str(role) for role in speaker_embeddings.keys())
    if any(not role for role in speaker_roles):
        raise ValueError("speaker roles must be non-empty strings.")
    audio_features = tuple(speaker_embeddings[role] for role in speaker_embeddings.keys())
    validate_matching_audio_embedding_shapes(
        *((f"speaker_embeddings[{role}]", emb) for role, emb in zip(speaker_roles, audio_features))
    )
    validate_audio_type(audio_type)

    speaker_count = len(audio_features)
    if speaker_count == 1:
        speaker_mode = "official_single"
        left_full_audio_emb = None
        full_audio_emb = audio_features[0]
    else:
        speaker_mode = "official_dual" if speaker_count == 2 else "advanced_experimental"
        left_full_audio_emb = audio_features[0]
        full_audio_emb = audio_features[1]

    payload = {
        "payload_type": AVATAR_AUDIO_PAYLOAD_TYPE,
        "audio_features": audio_features,
        "speaker_roles": speaker_roles,
        "speaker_mode": speaker_mode,
        "multi_audio_semantics": "parallel_overlay" if audio_type == AUDIO_TYPE_PARA else "sequential_add",
        "audio_encoder_type": audio_encoder_type,
        "audio_scale": float(audio_scale),
        "audio_cfg_scale": float(audio_cfg_scale),
        "audio_type": audio_type,
        "save_fps": int(save_fps),
        "generate_duration": calculate_generate_duration(int(save_fps), int(num_segments)),
        "target_output_frames": None if target_output_frames is None else int(target_output_frames),
        "full_audio_emb": full_audio_emb,
        "left_full_audio_emb": left_full_audio_emb,
        "back_full_audio_emb": None,
        "num_segments": int(num_segments),
        "audio_stride": int(audio_stride),
        "use_background_silent_audio": False,
        "masks": masks,
        "boxes": boxes or {},
    }
    validate_audio_conditioning_payload(payload)
    return payload


def validate_audio_conditioning_payload(au_cond: Any) -> None:
    if not isinstance(au_cond, Mapping):
        raise TypeError("Audio conditioning must be a mapping.")
    missing = [key for key in ("full_audio_emb", "num_segments", "audio_stride") if key not in au_cond]
    if missing:
        raise KeyError("Audio conditioning missing required keys: " + ", ".join(missing))

    if int(au_cond["num_segments"]) < 1:
        raise ValueError("Audio conditioning num_segments must be at least 1.")
    if int(au_cond["audio_stride"]) != AVATAR_AUDIO_STRIDE:
        raise ValueError(f"Avatar v1.5 audio_stride must be {AVATAR_AUDIO_STRIDE}.")
    if "payload_type" in au_cond:
        validate_avatar_audio_payload_metadata(au_cond)

    full_audio_emb = au_cond["full_audio_emb"]
    left_audio_emb = au_cond.get("left_full_audio_emb")
    back_audio_emb = au_cond.get("back_full_audio_emb")
    if left_audio_emb is None:
        validate_audio_embedding(full_audio_emb, "full_audio_emb")
        return

    embeddings = [
        ("left_full_audio_emb", left_audio_emb),
        ("full_audio_emb", full_audio_emb),
    ]
    if au_cond.get("use_background_silent_audio"):
        if back_audio_emb is None:
            raise ValueError("Background silent audio is enabled but back_full_audio_emb is missing.")
        embeddings.append(("back_full_audio_emb", back_audio_emb))
    elif back_audio_emb is not None:
        validate_audio_embedding(back_audio_emb, "back_full_audio_emb")
    validate_matching_audio_embedding_shapes(*embeddings)
