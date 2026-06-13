from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


MAX_STREAMING_PREFETCH_BLOCKS = 64
VAE_OFFLOAD_DEVICES = ("cpu", "cuda")
MPS_VISIBLE_VAE_OFFLOAD_DEVICES = ("cpu",)
OFFICIAL_DEFAULT_BASE_PRECISION = "bf16"
SUPPORTED_BASE_PRECISIONS = (OFFICIAL_DEFAULT_BASE_PRECISION,)
DISABLED_FP8_MODE = "disabled"
OFFICIAL_INT8_SHARDED_SOURCE = "official_int8_sharded"
MPS_EXPERIMENTAL_BRANCH = "macos-mps"
MPS_NON_MERGE_CONDITION = (
    "MPS inference work is restricted to the macos-mps branch. Do not merge MPS inference into main "
    "until model load, audio encode, attention fallback, VAE encode/decode, and end-to-end smoke "
    "inference evidence exists on Apple Silicon, followed by full validation and review."
)


@dataclass(frozen=True)
class AvatarRuntimePlan:
    device: str
    block_num: int
    streaming_prefetch_count: int | None
    move_dit_to_device: bool
    offload_dit_after_generate: bool
    vae_offload_device: str = "cpu"
    vae_to_device: bool = True
    lora_offload_owner: str = "model"


@dataclass(frozen=True)
class PrecisionRuntimePlan:
    base_precision: str
    fp8_mode: str
    gguf_model: str | None
    checkpoint_source: str
    quantization_source: str


@dataclass(frozen=True)
class MPSFeasibilityReport:
    target_branch: str
    supported: bool
    requested_device: str
    recommended_dtype: str
    attention_backend: str
    cuda_only_assumptions: tuple[str, ...]
    initialization_blockers: tuple[str, ...]
    non_merge_condition: str


def normalize_device_name(device: Any) -> str:
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None)
    if device_type:
        if device_index is None:
            return str(device_type).lower()
        return f"{str(device_type).lower()}:{device_index}"
    return str(device).lower()


def require_cuda_device(device: Any) -> str:
    normalized = normalize_device_name(device)
    if not normalized.startswith("cuda"):
        raise RuntimeError(
            "LongCat Avatar inference requires a CUDA device; "
            f"current device is '{normalized}'. CPU and MPS are not supported."
        )
    return normalized


def normalize_runtime_backend(device: Any) -> str:
    return normalize_device_name(device).split(":", 1)[0]


def require_supported_runtime_device(device: Any) -> str:
    normalized = normalize_device_name(device)
    backend = normalize_runtime_backend(normalized)
    if backend in {"cuda", "mps"}:
        return normalized
    raise RuntimeError(
        "LongCat Avatar inference requires a CUDA or MPS device; "
        f"current device is '{normalized}'. CPU is not supported."
    )


def normalize_block_num(block_num: Any) -> int:
    try:
        normalized = int(block_num)
    except (TypeError, ValueError) as exc:
        raise ValueError("block_num must be an integer.") from exc
    if normalized < 0 or normalized > MAX_STREAMING_PREFETCH_BLOCKS:
        raise ValueError(f"block_num must be between 0 and {MAX_STREAMING_PREFETCH_BLOCKS}; got {normalized}.")
    return normalized


def normalize_offload_device(offload_device: Any) -> str:
    normalized = str(offload_device).lower()
    if normalized not in VAE_OFFLOAD_DEVICES:
        raise ValueError(
            f"offload_device must be one of {', '.join(VAE_OFFLOAD_DEVICES)}; got {offload_device!r}."
        )
    return normalized


def resolve_vae_offload_device(offload_device: Any, runtime_device: Any) -> str:
    requested = normalize_offload_device(offload_device)
    if requested == "cpu":
        return "cpu"
    if normalize_runtime_backend(runtime_device) == "cuda":
        return "cuda"

    # IMPORTANT: stale MPS workflows may still carry offload_device=cuda.
    # CUDA here means "keep VAE resident on the active CUDA device"; on MPS it
    # must not preserve the CUDA label because downstream offload logic treats
    # it as a GPU-resident CUDA memory mode.
    return "cpu"


def build_runtime_plan(device: Any, block_num: Any, offload_device: Any = "cpu") -> AvatarRuntimePlan:
    normalized_device = require_supported_runtime_device(device)
    normalized_block = normalize_block_num(block_num)
    normalized_offload_device = resolve_vae_offload_device(offload_device, normalized_device)
    if normalize_runtime_backend(normalized_device) == "mps":
        streaming_prefetch_count = normalized_block if normalized_block > 0 else None
        eager_full_load = normalized_block == 0
        return AvatarRuntimePlan(
            device=normalized_device,
            block_num=normalized_block,
            streaming_prefetch_count=streaming_prefetch_count,
            move_dit_to_device=eager_full_load,
            offload_dit_after_generate=eager_full_load,
            vae_offload_device=normalized_offload_device,
        )
    streaming_prefetch_count = normalized_block if normalized_block > 0 else None
    eager_full_load = normalized_block == 0
    return AvatarRuntimePlan(
        device=normalized_device,
        block_num=normalized_block,
        streaming_prefetch_count=streaming_prefetch_count,
        move_dit_to_device=eager_full_load,
        offload_dit_after_generate=eager_full_load,
        vae_offload_device=normalized_offload_device,
    )


def _normalize_cuda_capability(cuda_capability: Any) -> tuple[int, int] | None:
    if cuda_capability is None:
        return None
    try:
        major, minor = cuda_capability
        return int(major), int(minor)
    except (TypeError, ValueError) as exc:
        raise ValueError("cuda_capability must be a (major, minor) pair.") from exc


def validate_precision_runtime_request(
    *,
    base_precision: str = OFFICIAL_DEFAULT_BASE_PRECISION,
    fp8_mode: str = DISABLED_FP8_MODE,
    gguf_model: str | None = None,
    checkpoint_source: str = "single_file_safetensors",
    cuda_capability: Any = None,
) -> PrecisionRuntimePlan:
    base_precision = str(base_precision).lower()
    fp8_mode = str(fp8_mode)
    if gguf_model not in (None, "", "none"):
        raise ValueError("GGUF DiT loading is not supported by this ComfyUI node yet.")
    if base_precision not in SUPPORTED_BASE_PRECISIONS:
        if base_precision == "fp16":
            raise NotImplementedError("FP16 runtime precision is not implemented for this release.")
        raise ValueError(
            f"Unsupported base precision '{base_precision}'. "
            f"Supported values: {SUPPORTED_BASE_PRECISIONS}."
        )
    if fp8_mode != DISABLED_FP8_MODE:
        capability = _normalize_cuda_capability(cuda_capability)
        if capability is not None and capability < (8, 9):
            raise ValueError("FP8 fast modes require CUDA compute capability 8.9 or newer.")
        raise NotImplementedError("FP8 runtime precision is not implemented for this release.")

    quantization_source = "none"
    if checkpoint_source == OFFICIAL_INT8_SHARDED_SOURCE:
        quantization_source = OFFICIAL_INT8_SHARDED_SOURCE
    return PrecisionRuntimePlan(
        base_precision=base_precision,
        fp8_mode=fp8_mode,
        gguf_model=None,
        checkpoint_source=checkpoint_source,
        quantization_source=quantization_source,
    )


def build_mps_feasibility_report(
    *,
    requested_device: Any = "mps",
    recommended_dtype: str = "fp16",
    attention_backend: str = "sdpa",
) -> MPSFeasibilityReport:
    normalized_device = normalize_device_name(requested_device)
    return MPSFeasibilityReport(
        target_branch=MPS_EXPERIMENTAL_BRANCH,
        supported=False,
        requested_device=normalized_device,
        recommended_dtype=str(recommended_dtype),
        attention_backend=str(attention_backend),
        cuda_only_assumptions=(
            "MPS positive block_num uses backend-aware layer streaming; CUDA stream recording remains CUDA-only.",
            "Embedded CUDA autocast, stream, and memory APIs must stay behind backend guards.",
            "MPS layer streaming uses synchronous layer moves and does not use CUDA streams or record_stream.",
            "Optional FlashAttention, xFormers, SageAttention, and Triton block-sparse paths are CUDA-oriented.",
        ),
        initialization_blockers=(
            "Minimal single-avatar MPS segment still lacks a first output artifact.",
            "Avatar DiT now passes the prior giant attention score-buffer allocation but stalls during "
            "denoising under MPS activation/TensorIterator memory pressure on the tested M2 host.",
            "End-to-end audio-driven frame timing, video decode memory behavior, and support wording remain unaccepted on MPS.",
        ),
        non_merge_condition=MPS_NON_MERGE_CONDITION,
    )


def apply_runtime_plan(model: Any, plan: AvatarRuntimePlan) -> None:
    if plan.vae_to_device:
        model.vae_to(plan.device)
    if plan.move_dit_to_device:
        model.to(plan.device)
    model.streaming_prefetch_count = plan.streaming_prefetch_count
    model.vae_offload_device = plan.vae_offload_device
    dit = getattr(model, "dit", None)
    if dit is not None:
        dit.lora_runtime_offload = plan.streaming_prefetch_count is not None


def cleanup_runtime_plan(
    model: Any,
    plan: AvatarRuntimePlan,
    *,
    empty_cache: Callable[[], None] | None = None,
) -> None:
    if plan.offload_dit_after_generate:
        model.to("cpu")
        dit = getattr(model, "dit", None)
        if dit is not None:
            dit.lora_runtime_offload = True
        if empty_cache is not None:
            empty_cache()
