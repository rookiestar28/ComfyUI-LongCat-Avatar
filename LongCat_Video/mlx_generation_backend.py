from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any

from .mlx_runner_contract import MlxRunnerRequest
from .mlx_runner_validation import MLX_VARIANT_DIRNAMES


@dataclass(frozen=True)
class MlxRuntimeModules:
    mx: Any
    nn: Any
    np: Any
    image: Any
    librosa: Any
    whisper_feature_extractor: Any
    t5_tokenizer_fast: Any
    imageio: Any
    autoencoder_cls: Any
    dit_cls: Any
    umt5_cls: Any
    whisper_cls: Any
    pipeline_cls: Any
    pipeline_config_cls: Any


def _load_runtime_modules() -> MlxRuntimeModules:
    try:
        import imageio
        import librosa
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from PIL import Image
        from transformers import T5TokenizerFast, WhisperFeatureExtractor

        from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
            LongCatVideoAvatarTransformer3DModel,
        )
        from longcat_video_avatar.models.umt5 import UMT5EncoderModel
        from longcat_video_avatar.models.whisper import WhisperEncoder
        from longcat_video_avatar.pipeline_mlx import LongCatAvatarPipeline, PipelineConfig
    except ImportError as exc:
        raise RuntimeError(
            "MLX generation dependencies are not installed in the runner environment. "
            "Install the isolated MLX runner dependencies before generation."
        ) from exc

    return MlxRuntimeModules(
        mx=mx,
        nn=nn,
        np=np,
        image=Image,
        librosa=librosa,
        whisper_feature_extractor=WhisperFeatureExtractor,
        t5_tokenizer_fast=T5TokenizerFast,
        imageio=imageio,
        autoencoder_cls=AutoencoderKLWan,
        dit_cls=LongCatVideoAvatarTransformer3DModel,
        umt5_cls=UMT5EncoderModel,
        whisper_cls=WhisperEncoder,
        pipeline_cls=LongCatAvatarPipeline,
        pipeline_config_cls=PipelineConfig,
    )


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path.name}.")
    return data


def _unique_shards(index_path: Path) -> tuple[str, ...]:
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Malformed sharded index: {index_path.name}.")
    return tuple(sorted({str(name) for name in weight_map.values()}))


def _quantize_dit_for_load(dit: Any, quant_cfg: dict[str, Any], runtime: MlxRuntimeModules) -> None:
    skip_patterns = quant_cfg.get(
        "skip_patterns",
        [
            "final_layer.linear",
            "t_embedder.",
            "y_embedder.",
            "adaLN_modulation.",
            "audio_adaLN_modulation.",
        ],
    )
    if not isinstance(skip_patterns, list):
        raise ValueError("DiT quantization skip_patterns must be a list.")

    def predicate(path: str, module: Any) -> bool:
        if not isinstance(module, runtime.nn.Linear):
            return False
        return not any(str(pattern) in path for pattern in skip_patterns)

    runtime.nn.quantize(
        dit,
        group_size=int(quant_cfg.get("group_size", 64)),
        bits=int(quant_cfg["bits"]),
        class_predicate=predicate,
    )


def build_mlx_pipeline(variant_dir: str | os.PathLike[str], runtime: MlxRuntimeModules | None = None) -> Any:
    runtime = runtime or _load_runtime_modules()
    root = Path(variant_dir).expanduser().resolve()

    vae = runtime.autoencoder_cls.from_config(_read_json(root / "vae" / "config.json"))
    vae.load_weights(str(root / "vae" / "diffusion_pytorch_model.safetensors"), strict=False)

    umt5 = runtime.umt5_cls.from_config(_read_json(root / "text_encoder" / "config.json"))
    for shard_name in _unique_shards(root / "text_encoder" / "model.safetensors.index.json"):
        umt5.load_weights(str(root / "text_encoder" / shard_name), strict=False)

    whisper = runtime.whisper_cls.from_config(_read_json(root / "audio_encoder" / "config.json"))
    whisper.load_weights(str(root / "audio_encoder" / "model.safetensors"), strict=False)

    dit_cfg = _read_json(root / "dit" / "config.json")
    quant_cfg = dit_cfg.get("quantization")
    dit = runtime.dit_cls.from_config(dit_cfg)
    if quant_cfg is not None:
        if not isinstance(quant_cfg, dict):
            raise ValueError("DiT quantization metadata must be an object.")
        _quantize_dit_for_load(dit, quant_cfg, runtime)
    for shard_name in _unique_shards(root / "dit" / "diffusion_pytorch_model.safetensors.index.json"):
        dit.load_weights(str(root / "dit" / shard_name), strict=False)

    runtime.mx.eval(vae.parameters(), umt5.parameters(), whisper.parameters(), dit.parameters())
    return runtime.pipeline_cls(
        vae=vae,
        text_encoder=umt5,
        audio_encoder=whisper,
        dit=dit,
        config=runtime.pipeline_config_cls(),
    )


def _preprocess_image(image_path: str | os.PathLike[str], *, height: int, width: int, runtime: MlxRuntimeModules) -> Any:
    image = runtime.image.open(image_path).convert("RGB").resize((width, height), runtime.image.BICUBIC)
    arr = runtime.np.asarray(image, dtype=runtime.np.float32) / 127.5 - 1.0
    arr = arr.transpose(2, 0, 1)
    return runtime.mx.array(arr[None, :, None, :, :])


def _preprocess_audio_mel(audio_path: str | os.PathLike[str], runtime: MlxRuntimeModules) -> Any:
    audio, _ = runtime.librosa.load(str(audio_path), sr=16000)
    extractor = runtime.whisper_feature_extractor.from_pretrained("openai/whisper-large-v3")
    inputs = extractor(audio, sampling_rate=16000, return_tensors="np")
    return runtime.mx.array(inputs.input_features.astype(runtime.np.float32))


def _tokenize_prompt(prompt: str, tokenizer_dir: Path, runtime: MlxRuntimeModules) -> tuple[Any, Any]:
    tokenizer = runtime.t5_tokenizer_fast.from_pretrained(str(tokenizer_dir))
    encoded = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=512, truncation=True)
    return runtime.mx.array(encoded.input_ids), runtime.mx.array(encoded.attention_mask)


def _runtime_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def run_mlx_generation(
    request: MlxRunnerRequest,
    *,
    variant_dir: str | os.PathLike[str] | None = None,
    runtime: MlxRuntimeModules | None = None,
) -> dict[str, Any]:
    runtime = runtime or _load_runtime_modules()
    root = Path(variant_dir).expanduser().resolve() if variant_dir is not None else (
        Path(request.weights_root).expanduser().resolve() / MLX_VARIANT_DIRNAMES[request.variant]
    )
    output_root = Path(request.output_dir).expanduser().resolve()
    frames_path = output_root / f"{request.output_basename}.npy"
    video_path = output_root / f"{request.output_basename}.mp4"

    timings: dict[str, float] = {}
    warnings: list[str] = []

    started = time.perf_counter()
    pipeline = build_mlx_pipeline(root, runtime=runtime)
    timings["load_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    image = _preprocess_image(request.image_path, height=request.height, width=request.width, runtime=runtime)
    audio_mel = _preprocess_audio_mel(request.audio_path, runtime=runtime)
    tokenizer_dir = root / "tokenizer"
    ids, mask = _tokenize_prompt(request.prompt, tokenizer_dir, runtime)
    text_hidden = pipeline.text_encoder(ids, mask=mask)
    text_embeds = text_hidden[:, None, :, :]
    text_mask = mask[:, None, None, :]
    empty_ids, empty_mask = _tokenize_prompt(request.negative_prompt, tokenizer_dir, runtime)
    uncond_hidden = pipeline.text_encoder(empty_ids, mask=empty_mask)
    uncond_embeds = uncond_hidden[:, None, :, :]
    uncond_mask = empty_mask[:, None, None, :]
    timings["preprocess_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    video = pipeline(
        image=image,
        audio_mel=audio_mel,
        text_embeds=text_embeds,
        text_mask=text_mask,
        uncond_embeds=uncond_embeds,
        uncond_mask=uncond_mask,
        height=request.height,
        width=request.width,
        num_frames=request.num_frames,
        seed=request.seed,
    )
    runtime.mx.eval(video)
    timings["inference_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    frames = (
        (runtime.np.asarray(video).transpose(0, 2, 3, 4, 1)[0] * 127.5 + 127.5)
        .clip(0, 255)
        .astype(runtime.np.uint8)
    )
    runtime.np.save(str(frames_path), frames)
    try:
        writer = runtime.imageio.get_writer(str(video_path), fps=request.fps, codec="libx264", quality=8)
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
    except Exception as exc:
        warnings.append(f"mp4 export failed; frame artifact retained: {exc.__class__.__name__}")
        video_path = Path("")
    timings["export_seconds"] = time.perf_counter() - started

    return {
        "video_path": str(video_path) if video_path else "",
        "frames_path": str(frames_path),
        "timings": timings,
        "runtime": {
            "backend": "mlx",
            "mlx": _runtime_version("mlx"),
            "variant": request.variant,
        },
        "warnings": tuple(warnings),
    }


__all__ = [
    "MlxRuntimeModules",
    "build_mlx_pipeline",
    "run_mlx_generation",
]
