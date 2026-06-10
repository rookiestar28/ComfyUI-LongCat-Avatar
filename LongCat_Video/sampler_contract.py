from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .video_output import normalize_mux_audio_path


MAX_SEED = 2**31 - 1
SINGLE_MODE = "single"
MULTI_MODE = "multi"
STAGE_AT2V = "at2v"
STAGE_AI2V = "ai2v"
SUPPORTED_SINGLE_STAGES = (STAGE_AT2V, STAGE_AI2V)
SUPPORTED_MULTI_STAGES = (STAGE_AI2V,)
RESOLUTION_DIMENSIONS = {
    "480p": (480, 832),
    "720p": (768, 1280),
}
NUM_FRAMES = 93
NUM_COND_FRAMES = 13
IMAGE_CHANNELS = (3, 4)
SUPPORTED_INSUFFICIENT_AUDIO_POLICIES = ("clamp", "mirror_from_end")


@dataclass(frozen=True)
class AudioWindowSpec:
    frames_processed: int
    num_frames: int
    overlap: int
    audio_stride: int
    audio_start_idx: int
    audio_end_idx: int
    center_indices: tuple[tuple[int, ...], ...]
    if_not_enough_audio: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "num_frames": self.num_frames,
            "overlap": self.overlap,
            "audio_stride": self.audio_stride,
            "audio_start_idx": self.audio_start_idx,
            "audio_end_idx": self.audio_end_idx,
            "center_indices": self.center_indices,
            "if_not_enough_audio": self.if_not_enough_audio,
        }


@dataclass(frozen=True)
class LatentBookkeepingSpec:
    ref_latent_indices: tuple[int, ...]
    adjusted_clean_latent_indices: tuple[int, ...]
    generated_latent_count: int
    num_ref_latents: int
    num_cond_latents: int


@dataclass(frozen=True)
class SamplerExecutionRequest:
    mode: str
    stage_1: str
    resolution: str
    seed: int
    steps: int
    text_guidance_scale: float
    audio_guidance_scale: float
    ref_img_index: int
    mask_frame_range: int
    block_num: int
    mux_audio_path: str
    offload_device: str


def select_sampler_mode(audio_conditioning: Mapping[str, Any]) -> str:
    return MULTI_MODE if audio_conditioning.get("left_full_audio_emb") is not None else SINGLE_MODE


def classify_audio_payload_window_state(audio_conditioning: Mapping[str, Any]) -> str:
    audio_emb_slice = audio_conditioning.get("audio_emb_slice")
    if audio_emb_slice:
        return "sliced"
    if audio_conditioning.get("audio_slice_indices"):
        return "indexed"
    if audio_conditioning.get("audio_features") or audio_conditioning.get("full_audio_emb") is not None:
        return "full"
    raise ValueError("Audio conditioning does not contain full or sliced audio payload data.")


def build_latent_bookkeeping_spec(
    *,
    clean_latent_indices: tuple[int, ...] = (),
    ref_latent_count: int = 0,
    generated_latent_count: int,
) -> LatentBookkeepingSpec:
    ref_latent_count = int(ref_latent_count)
    generated_latent_count = int(generated_latent_count)
    if ref_latent_count < 0:
        raise ValueError("ref_latent_count must be non-negative.")
    if generated_latent_count < 0:
        raise ValueError("generated_latent_count must be non-negative.")

    normalized_clean = tuple(int(idx) for idx in clean_latent_indices)
    for idx in normalized_clean:
        if idx < 0 or idx >= generated_latent_count:
            raise ValueError("clean_latent_indices must point inside generated latents.")

    ref_indices = tuple(range(ref_latent_count))
    adjusted_generated = tuple(idx + ref_latent_count for idx in normalized_clean)
    adjusted = ref_indices + adjusted_generated
    return LatentBookkeepingSpec(
        ref_latent_indices=ref_indices,
        adjusted_clean_latent_indices=adjusted,
        generated_latent_count=generated_latent_count,
        num_ref_latents=ref_latent_count,
        num_cond_latents=len(adjusted),
    )


def resolve_resolution_dimensions(resolution: str) -> tuple[int, int]:
    try:
        return RESOLUTION_DIMENSIONS[resolution]
    except KeyError as exc:
        raise ValueError(f"Unsupported resolution '{resolution}'. Expected one of {tuple(RESOLUTION_DIMENSIONS)}.") from exc


def normalize_seed(seed: int) -> int:
    normalized = int(seed)
    if normalized < 0 or normalized > MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}; got {normalized}.")
    return normalized


def validate_continuation_parameters(ref_img_index: int, mask_frame_range: int) -> tuple[int, int]:
    ref_img_index = int(ref_img_index)
    mask_frame_range = int(mask_frame_range)
    if ref_img_index < 0:
        raise ValueError("ref_img_index must be non-negative.")
    if mask_frame_range < 0:
        raise ValueError("mask_frame_range must be non-negative.")
    return ref_img_index, mask_frame_range


def validate_sampler_stage(mode: str, stage_1: str) -> None:
    if mode == SINGLE_MODE and stage_1 not in SUPPORTED_SINGLE_STAGES:
        raise ValueError(f"Single-audio mode supports stages {SUPPORTED_SINGLE_STAGES}; got '{stage_1}'.")
    if mode == MULTI_MODE and stage_1 not in SUPPORTED_MULTI_STAGES:
        raise ValueError("Multi-audio mode is image-conditioned and supports only stage_1='ai2v'.")
    if mode not in (SINGLE_MODE, MULTI_MODE):
        raise ValueError(f"Unsupported sampler mode '{mode}'.")


def validate_sampler_inputs(
    audio_conditioning: Mapping[str, Any],
    *,
    stage_1: str,
    resolution: str,
    seed: int,
    ref_img_index: int,
    mask_frame_range: int,
) -> str:
    mode = select_sampler_mode(audio_conditioning)
    validate_sampler_stage(mode, stage_1)
    resolve_resolution_dimensions(resolution)
    normalize_seed(seed)
    validate_continuation_parameters(ref_img_index, mask_frame_range)
    return mode


def build_sampler_execution_request(
    audio_conditioning: Mapping[str, Any],
    *,
    stage_1: str,
    resolution: str,
    seed: int,
    steps: int,
    text_guidance_scale: float,
    audio_guidance_scale: float,
    ref_img_index: int,
    mask_frame_range: int,
    block_num: int,
    mux_audio_path: str | None,
    offload_device: str,
) -> SamplerExecutionRequest:
    mode = validate_sampler_inputs(
        audio_conditioning,
        stage_1=stage_1,
        resolution=resolution,
        seed=seed,
        ref_img_index=ref_img_index,
        mask_frame_range=mask_frame_range,
    )
    normalized_ref_img_index, normalized_mask_frame_range = validate_continuation_parameters(
        ref_img_index,
        mask_frame_range,
    )
    return SamplerExecutionRequest(
        mode=mode,
        stage_1=str(stage_1),
        resolution=str(resolution),
        seed=normalize_seed(seed),
        steps=int(steps),
        text_guidance_scale=float(text_guidance_scale),
        audio_guidance_scale=float(audio_guidance_scale),
        ref_img_index=normalized_ref_img_index,
        mask_frame_range=normalized_mask_frame_range,
        block_num=int(block_num),
        mux_audio_path=normalize_mux_audio_path(mux_audio_path),
        offload_device=str(offload_device),
    )


def segment_audio_start(segment_idx: int, audio_stride: int = 1) -> int:
    segment_idx = int(segment_idx)
    audio_stride = int(audio_stride)
    if segment_idx < 0:
        raise ValueError("segment_idx must be non-negative.")
    if audio_stride < 1:
        raise ValueError("audio_stride must be at least 1.")
    return segment_idx * audio_stride * (NUM_FRAMES - NUM_COND_FRAMES)


def _normalize_audio_index(index: int, full_audio_frames: int, policy: str) -> int:
    if full_audio_frames < 1:
        raise ValueError("full_audio_frames must be at least 1.")
    if policy == "clamp":
        return max(0, min(index, full_audio_frames - 1))
    if policy == "mirror_from_end":
        while index < 0 or index >= full_audio_frames:
            if index < 0:
                index = -index
            if index >= full_audio_frames:
                index = (full_audio_frames - 1) - (index - (full_audio_frames - 1))
        return max(0, min(index, full_audio_frames - 1))
    raise ValueError(f"Unsupported if_not_enough_audio policy '{policy}'.")


def build_audio_window_spec(
    *,
    frames_processed: int = 0,
    num_frames: int = NUM_FRAMES,
    overlap: int = NUM_COND_FRAMES,
    audio_stride: int = 1,
    full_audio_frames: int | None = None,
    if_not_enough_audio: str = "clamp",
    context_radius: int = 2,
) -> AudioWindowSpec:
    frames_processed = int(frames_processed)
    num_frames = int(num_frames)
    overlap = int(overlap)
    audio_stride = int(audio_stride)
    context_radius = int(context_radius)
    if frames_processed < 0:
        raise ValueError("frames_processed must be non-negative.")
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if audio_stride < 1:
        raise ValueError("audio_stride must be at least 1.")
    if context_radius < 0:
        raise ValueError("context_radius must be non-negative.")
    if if_not_enough_audio not in SUPPORTED_INSUFFICIENT_AUDIO_POLICIES:
        raise ValueError(
            f"Unsupported if_not_enough_audio policy '{if_not_enough_audio}'. "
            f"Expected one of {SUPPORTED_INSUFFICIENT_AUDIO_POLICIES}."
        )

    start_frame = 0 if frames_processed == 0 else max(frames_processed - overlap, 0)
    audio_start_idx = start_frame * audio_stride
    audio_end_idx = audio_start_idx + num_frames * audio_stride
    centers = range(audio_start_idx, audio_end_idx, audio_stride)
    offsets = range(-context_radius, context_radius + 1)
    if full_audio_frames is None:
        center_indices = tuple(tuple(center + offset for offset in offsets) for center in centers)
    else:
        center_indices = tuple(
            tuple(
                _normalize_audio_index(center + offset, int(full_audio_frames), if_not_enough_audio)
                for offset in offsets
            )
            for center in centers
        )

    return AudioWindowSpec(
        frames_processed=frames_processed,
        num_frames=num_frames,
        overlap=overlap,
        audio_stride=audio_stride,
        audio_start_idx=audio_start_idx,
        audio_end_idx=audio_end_idx,
        center_indices=center_indices,
        if_not_enough_audio=if_not_enough_audio,
    )


def _first_audio_feature_frame_count(audio_payload: Mapping[str, Any]) -> int:
    features = audio_payload.get("audio_features")
    if not isinstance(features, tuple) or not features:
        full_audio_emb = audio_payload.get("full_audio_emb")
        features = (full_audio_emb,) if full_audio_emb is not None else ()
    if not features:
        raise ValueError("Audio payload has no audio features to slice.")
    shape = _shape_tuple(features[0], "audio feature")
    return shape[0]


def _materialize_audio_slice(embedding: Any, center_indices: tuple[tuple[int, ...], ...]) -> Any:
    getitem = getattr(embedding, "__getitem__", None)
    if getitem is None:
        return None
    try:
        import torch

        device = getattr(embedding, "device", None)
        index_tensor = torch.tensor(center_indices, dtype=torch.long, device=device)
        return embedding[index_tensor]
    except Exception:
        try:
            return embedding[center_indices]
        except Exception:
            return None


def build_audio_window_payload(
    audio_payload: Mapping[str, Any],
    *,
    frames_processed: int = 0,
    num_frames: int = NUM_FRAMES,
    overlap: int = NUM_COND_FRAMES,
    if_not_enough_audio: str = "clamp",
    ref_img_index: int = 10,
    mask_frame_range: int = 3,
    prev_latents: Any | None = None,
    prev_images: Any | None = None,
    vae: Any | None = None,
    samples: Any | None = None,
) -> dict[str, Any]:
    if (prev_images is None) != (vae is None):
        raise ValueError("prev_images and vae must be provided together for VAE overlap re-encode.")
    validate_continuation_parameters(ref_img_index, mask_frame_range)
    audio_stride = int(audio_payload.get("audio_stride", 1))
    full_audio_frames = _first_audio_feature_frame_count(audio_payload)
    spec = build_audio_window_spec(
        frames_processed=frames_processed,
        num_frames=num_frames,
        overlap=overlap,
        audio_stride=audio_stride,
        full_audio_frames=full_audio_frames,
        if_not_enough_audio=if_not_enough_audio,
    )

    features = audio_payload.get("audio_features")
    if not isinstance(features, tuple) or not features:
        features = (audio_payload["full_audio_emb"],)
    slices = tuple(
        result
        for result in (_materialize_audio_slice(feature, spec.center_indices) for feature in features)
        if result is not None
    )

    overlap_source = "none"
    if overlap > 0 and prev_images is not None and vae is not None:
        overlap_source = "vae_reencode_available"
    elif overlap > 0 and prev_latents is not None:
        overlap_source = "latent_overlap_available"

    window_payload = dict(audio_payload)
    window_payload.update(
        {
            "payload_type": "longcat_avatar_audio_window",
            "source_payload_type": audio_payload.get("payload_type"),
            "window": spec.as_dict(),
            "audio_slice_indices": spec.center_indices,
            "audio_emb_slice": slices,
            "ref_img_index": int(ref_img_index),
            "mask_frame_range": int(mask_frame_range),
            "overlap_source": overlap_source,
        }
    )
    if samples is not None:
        window_payload["samples_slice"] = samples
    return window_payload


def expected_output_frames(num_segments: int) -> int:
    num_segments = int(num_segments)
    if num_segments < 1:
        raise ValueError("num_segments must be at least 1.")
    return NUM_FRAMES + (num_segments - 1) * (NUM_FRAMES - NUM_COND_FRAMES)


def _shape_tuple(value: Any, role: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{role} must be a tensor-like object with a shape.")
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{role} has an invalid shape.") from exc


def validate_output_tensor_contract(value: Any, *, expected_frames: int | None = None) -> tuple[int, int, int, int]:
    shape = _shape_tuple(value, "output image tensor")
    if len(shape) != 4:
        raise ValueError(f"output image tensor must have rank 4 [frames, height, width, channels]; got rank {len(shape)}.")
    frames, height, width, channels = shape
    if expected_frames is not None and frames != int(expected_frames):
        raise ValueError(f"output image tensor frame count must be {expected_frames}; got {frames}.")
    if frames < 1 or height < 1 or width < 1:
        raise ValueError("output image tensor frames, height, and width must be positive.")
    if channels not in IMAGE_CHANNELS:
        raise ValueError(f"output image tensor channel count must be one of {IMAGE_CHANNELS}; got {channels}.")
    return shape


def trim_output_tensor_to_target_frames(value: Any, target_frames: int | None) -> Any:
    if target_frames is None:
        return value
    target_frames = int(target_frames)
    if target_frames < 1:
        raise ValueError("target_output_frames must be at least 1.")
    frames, _, _, _ = validate_output_tensor_contract(value)
    if target_frames > frames:
        raise ValueError(
            f"target_output_frames exceeds generated frame count; target {target_frames}, generated {frames}."
        )
    if target_frames == frames:
        return value
    getitem = getattr(value, "__getitem__", None)
    if getitem is None:
        raise TypeError("output image tensor must support slicing for target frame trimming.")
    trimmed = value[:target_frames]
    validate_output_tensor_contract(trimmed, expected_frames=target_frames)
    return trimmed
