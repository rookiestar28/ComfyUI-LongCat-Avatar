 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os

from comfy_api.latest import  io, ui
import folder_paths
from .node_utils import  clear_comfyui_cache,get_runtime_device,tensor2image,audio2path
from .LongCat_Video.run_demo_avatar_single_audio_to_video import load_longcat_video_model,generate,get_audio_vocal,get_audio_emb,load_audio_vocal
from .LongCat_Video.run_demo_avatar_multi_audio_to_video import generate_multi
from .LongCat_Video.audio_contract import (
    validate_audio_conditioning_payload,
    validate_longcat_avatar_whisper_model_name,
)
from .LongCat_Video.backend_capabilities import empty_cache as backend_empty_cache
from .LongCat_Video.audio_crop import crop_audio_payload
from .LongCat_Video.backend_dtype_policy import resolve_backend_dtype_policy
from .LongCat_Video.bbox_contract import parse_person_boxes
from .LongCat_Video.model_contract import (
    AVATAR_MAX_GUIDANCE_SCALE,
    AVATAR_MAX_INFERENCE_STEPS,
    AVATAR_MIN_GUIDANCE_SCALE,
    AVATAR_MIN_INFERENCE_STEPS,
    OFFICIAL_V15_DISTILL_AUDIO_CFG,
    OFFICIAL_V15_DISTILL_STEPS,
    OFFICIAL_V15_DISTILL_TEXT_CFG,
)
from .LongCat_Video.performance_contract import (
    VAE_OFFLOAD_DEVICES,
    apply_runtime_plan,
    build_runtime_plan,
    cleanup_runtime_plan,
)
from .LongCat_Video.sampler_contract import build_audio_window_payload, build_sampler_execution_request
from .LongCat_Video.text_conditioning import (
    DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT,
    TEXT_CONDITIONING_SOURCE_CLIP,
    TEXT_ENCODER_OFFLOAD_DEVICES,
    encode_official_text_conditioning,
    extract_scheduled_text_embedding,
    resolve_or_download_official_text_encoder_layout,
    validate_text_conditioning_payload,
)
from .LongCat_Video.video_output import save_muxed_video, validate_mux_audio_path
from .LongCat_Video.attention_contract import ATTENTION_MODES, validate_attention_mode_for_device
from .LongCat_Video.checkpoint_contract import (
    OFFICIAL_INT8_SHARDED,
    OFFICIAL_SHARDED,
    SINGLE_FILE_SAFETENSORS,
    build_download_manifest,
    download_missing_checkpoint_assets,
    inspect_checkpoint_source,
)
from .LongCat_Video.debug_profile import LongCatDebugProfiler

MAX_SEED = np.iinfo(np.int32).max
node_longcat_path = os.path.dirname(os.path.abspath(__file__))
weigths_longcat_current_path = os.path.join(folder_paths.models_dir, "longcat")
if not os.path.exists(weigths_longcat_current_path):
    os.makedirs(weigths_longcat_current_path)
folder_paths.add_model_folder_path("longcat", weigths_longcat_current_path) #  longcat dir

INFERENCE_WEIGHT_MODES = [
    SINGLE_FILE_SAFETENSORS,
    OFFICIAL_SHARDED,
    OFFICIAL_INT8_SHARDED,
]
DEFAULT_SINGLE_FILE_DIT = "LongCat-Video-Avatar-1.5-int8.safetensors"
DEFAULT_OFFICIAL_CHECKPOINTS = {
    OFFICIAL_SHARDED: "LongCat-Video-Avatar-1.5/base_model/diffusion_pytorch_model.safetensors.index.json",
    OFFICIAL_INT8_SHARDED: "LongCat-Video-Avatar-1.5/base_model_int8/quantized_model.safetensors.index.json",
}


def _resolve_single_file_model_name(inference_weight_mode):
    if inference_weight_mode == SINGLE_FILE_SAFETENSORS:
        return DEFAULT_SINGLE_FILE_DIT
    if inference_weight_mode in DEFAULT_OFFICIAL_CHECKPOINTS:
        return None
    raise ValueError(f"Unsupported inference_weight_mode: {inference_weight_mode}")


def _resolve_official_checkpoint_name(inference_weight_mode):
    if inference_weight_mode == SINGLE_FILE_SAFETENSORS:
        return "none"
    try:
        return DEFAULT_OFFICIAL_CHECKPOINTS[inference_weight_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported inference_weight_mode: {inference_weight_mode}") from exc


def _resolve_official_checkpoint_path(
    inference_weight_mode,
    auto_download_missing_weights,
):
    official_checkpoint = _resolve_official_checkpoint_name(inference_weight_mode)
    selected_path = (
        folder_paths.get_full_path("longcat", official_checkpoint)
        if official_checkpoint != "none"
        else None
    )
    if selected_path is not None:
        inspection = inspect_checkpoint_source(inference_weight_mode, selected_path)
        if inspection.is_complete:
            return selected_path
        if not auto_download_missing_weights:
            raise FileNotFoundError(
                "Official checkpoint is incomplete; missing: "
                + ", ".join(inspection.missing_files)
            )

    if not auto_download_missing_weights:
        raise FileNotFoundError(
            "No complete official checkpoint was selected. Select an official "
            "checkpoint index under models/longcat or enable auto_download_missing_weights."
        )

    manifest = build_download_manifest(inference_weight_mode, weigths_longcat_current_path)
    download_missing_checkpoint_assets(manifest)
    inspection = inspect_checkpoint_source(inference_weight_mode, manifest.local_dir)
    if not inspection.is_complete:
        raise FileNotFoundError(
            "Official checkpoint download did not produce a complete checkpoint; missing: "
            + ", ".join(inspection.missing_files)
        )
    return manifest.local_dir


class LongCat_Video_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_Model",
            display_name="(auto)Load LongCat Avatar Model",
            category="LongCat Avatar",
            inputs=[
                io.Combo.Input("inference_weight_mode",options=INFERENCE_WEIGHT_MODES),
                io.Combo.Input("attention_mode",options=ATTENTION_MODES),
                io.Boolean.Input("auto_download_missing_weights", default=True),
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae")),
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras")),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, inference_weight_mode,attention_mode,auto_download_missing_weights,vae,lora) -> io.NodeOutput:
        validate_attention_mode_for_device(attention_mode, get_runtime_device())
        clear_comfyui_cache()
        diffusion_models = _resolve_single_file_model_name(inference_weight_mode)
        dit_path=folder_paths.get_full_path("diffusion_models",diffusion_models) if diffusion_models is not None else None
        vae_path=folder_paths.get_full_path("vae",vae) if vae != "none" else None
        lora_path=folder_paths.get_full_path("loras",lora) if lora != "none" else None
        if inference_weight_mode in (OFFICIAL_SHARDED, OFFICIAL_INT8_SHARDED):
            official_checkpoint_path = _resolve_official_checkpoint_path(
                inference_weight_mode,
                bool(auto_download_missing_weights),
            )
            model=load_longcat_video_model(
                None,
                vae_path,
                lora_path,
                node_longcat_path,
                use_int8=inference_weight_mode == OFFICIAL_INT8_SHARDED,
                checkpoint_source=inference_weight_mode,
                official_checkpoint_path=official_checkpoint_path,
                attention_mode=attention_mode,
            )
        else:
            model_path=dit_path
            selected_model_name = os.path.basename(model_path).lower() if model_path else ""
            use_int8 = selected_model_name.endswith(".safetensors") and "int8" in selected_model_name
            model=load_longcat_video_model(
                model_path,
                vae_path,
                lora_path,
                node_longcat_path,
                use_int8=use_int8,
                checkpoint_source=inference_weight_mode,
                attention_mode=attention_mode,
            )
        return io.NodeOutput(model)


class LongCat_Video_SM_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_Sampler",
            display_name="LongCat Avatar Sampler",
            category="LongCat Avatar",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("te_cond"),
                io.Conditioning.Input("au_cond"),
                io.Image.Input("image"),
                io.Combo.Input("stage_1",options= ['ai2v', 'at2v']),
                io.Combo.Input("resolution",options= ['480p', '720p']),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("steps", default=OFFICIAL_V15_DISTILL_STEPS, min=AVATAR_MIN_INFERENCE_STEPS, max=AVATAR_MAX_INFERENCE_STEPS, step=1),
                io.Float.Input("text_guidance_scale", default=OFFICIAL_V15_DISTILL_TEXT_CFG, min=AVATAR_MIN_GUIDANCE_SCALE, max=AVATAR_MAX_GUIDANCE_SCALE, step=0.1, ),
                io.Float.Input("audio_guidance_scale", default=OFFICIAL_V15_DISTILL_AUDIO_CFG, min=AVATAR_MIN_GUIDANCE_SCALE, max=AVATAR_MAX_GUIDANCE_SCALE, step=0.1,),
                io.Int.Input("ref_img_index", default=10, min=0, max=1024, step=1),
                io.Int.Input("mask_frame_range", default=3, min=0, max=1024, step=1),
                io.Int.Input("block_num", default=1, min=0, max=64,step=1),
                io.String.Input("mux_audio_path", default="", multiline=False),
                io.Combo.Input("offload_device", options=list(VAE_OFFLOAD_DEVICES), default="cpu"),
                io.Boolean.Input("debug_mode", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="video_path"),
            ],
        )

    @classmethod
    def execute(cls, model,te_cond,au_cond,image,stage_1,resolution, seed, steps,text_guidance_scale,audio_guidance_scale,ref_img_index,mask_frame_range,block_num,mux_audio_path,offload_device="cpu",debug_mode=False,) -> io.NodeOutput:
        runtime_device = get_runtime_device()
        debug_profile = LongCatDebugProfiler(bool(debug_mode), label="sampler", device=runtime_device)
        with debug_profile.phase("clear_comfyui_cache"):
            clear_comfyui_cache()
        with debug_profile.phase("validate_inputs"):
            request = build_sampler_execution_request(
                au_cond,
                stage_1=stage_1,
                resolution=resolution,
                seed=seed,
                steps=steps,
                text_guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                ref_img_index=ref_img_index,
                mask_frame_range=mask_frame_range,
                block_num=block_num,
                mux_audio_path=mux_audio_path,
                offload_device=offload_device,
            )
            if request.mux_audio_path:
                validate_mux_audio_path(request.mux_audio_path)
            validate_text_conditioning_payload(te_cond, require_negative=True)
            validate_audio_conditioning_payload(au_cond)
        with debug_profile.phase("build_runtime_plan"):
            runtime_plan = build_runtime_plan(runtime_device, request.block_num, request.offload_device)
            print(
                "[INFO] LongCat runtime plan: "
                f"block_num={getattr(runtime_plan, 'block_num', request.block_num)}, "
                f"streaming_prefetch_count={getattr(runtime_plan, 'streaming_prefetch_count', 'unknown')}, "
                f"move_dit_to_device={getattr(runtime_plan, 'move_dit_to_device', 'unknown')}, "
                f"offload_dit_after_generate={getattr(runtime_plan, 'offload_dit_after_generate', 'unknown')}, "
                f"vae_offload_device={getattr(runtime_plan, 'vae_offload_device', request.offload_device)}"
            )
        with debug_profile.phase("apply_runtime_plan"):
            apply_runtime_plan(model, runtime_plan)
        try:
            with debug_profile.phase("prepare_image"):
                cond_image = tensor2image(image)
            with debug_profile.phase("generate", mode=request.mode):
                if request.mode == "multi":
                    image=generate_multi(model,au_cond,te_cond,runtime_device,request.seed,cond_image,request.resolution,
                        request.text_guidance_scale,request.audio_guidance_scale,request.steps,request.ref_img_index,request.mask_frame_range,model.use_distill,
                        debug_profile=debug_profile.child("multi"))
                else:
                    image=generate(model,au_cond,te_cond,runtime_device,request.seed,request.stage_1,cond_image,request.resolution,
                        request.text_guidance_scale,request.audio_guidance_scale,request.steps,request.ref_img_index,request.mask_frame_range,
                        model.use_distill,debug_profile=debug_profile.child("single"))
        finally:
            with debug_profile.phase("cleanup_runtime_plan"):
                cleanup_runtime_plan(
                    model,
                    runtime_plan,
                    empty_cache=lambda: backend_empty_cache(runtime_plan.device, torch_module=torch),
                )
        with debug_profile.phase("save_muxed_video", enabled=bool(request.mux_audio_path), mode=request.mode):
            video_path = save_muxed_video(
                image,
                enabled=bool(request.mux_audio_path),
                output_dir=folder_paths.get_output_directory(),
                audio_path=request.mux_audio_path,
                prefix=None,
                mode=request.mode,
            )
        return io.NodeOutput(image, video_path)

class LongCat_Video_SM_Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_Encode",
            display_name="LongCat Avatar Text Encode",
            category="LongCat Avatar",
            inputs=[
                io.Clip.Input("clip", optional=True),
                io.String.Input("text_encoder_root", default=DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT, multiline=False),
                io.Boolean.Input("auto_download_missing_text_encoder", default=True),
                io.Combo.Input("offload_device", options=list(TEXT_ENCODER_OFFLOAD_DEVICES), default="cpu"),
                io.String.Input("prompt",default="A western man stands on stage under dramatic lighting, holding a microphone close to their mouth. Wearing a vibrant red jacket with gold embroidery, the singer is speaking while smoke swirls around them, creating a dynamic and atmospheric scene.",multiline=True),
                io.String.Input("negative_prompt",default="Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards.",multiline=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="te_cond"),
                ],
        )
    @classmethod
    def execute(
        cls,
        clip=None,
        text_encoder_root=DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT,
        auto_download_missing_text_encoder=True,
        offload_device="cpu",
        prompt="",
        negative_prompt="",
    ) -> io.NodeOutput:
        runtime_device = get_runtime_device()
        dtype_policy = resolve_backend_dtype_policy(runtime_device, torch_module=torch)
        if clip is None:
            # CRITICAL: official path is preferred; connected legacy CLIP input deliberately selects single-file UMT5 fallback.
            layout = resolve_or_download_official_text_encoder_layout(
                text_encoder_root,
                weigths_longcat_current_path,
                auto_download_missing_text_encoder=bool(auto_download_missing_text_encoder),
            )
            te_cond = encode_official_text_conditioning(
                layout=layout,
                prompt=prompt,
                negative_prompt=negative_prompt,
                device=runtime_device,
                offload_device=offload_device,
                dtype=dtype_policy.text_encoder_dtype,
            )
            clear_comfyui_cache()
            return io.NodeOutput(te_cond)

        tokens = clip.tokenize(prompt)
        prompt_embeds=clip.encode_from_tokens_scheduled(tokens)
        prompt_embeds=extract_scheduled_text_embedding(prompt_embeds, "prompt_embeds")
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(1, 1, seq_len, -1)
        tokens = clip.tokenize(negative_prompt)
        negative_prompt_embeds=clip.encode_from_tokens_scheduled(tokens)
        negative_prompt_embeds=extract_scheduled_text_embedding(negative_prompt_embeds, "negative_prompt_embeds")
        _, seq_len, _ = negative_prompt_embeds.shape
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(1, 1, seq_len, -1)
        te_cond={
            "prompt_embeds":prompt_embeds.to(runtime_device,dtype_policy.text_encoder_dtype),
            "negative_prompt_embeds":negative_prompt_embeds.to(runtime_device,dtype_policy.text_encoder_dtype),
            "text":[prompt,negative_prompt],
            "conditioning_source": TEXT_CONDITIONING_SOURCE_CLIP,
        } # #torch.Size([1, 1, 512, 4096])
        validate_text_conditioning_payload(te_cond, require_negative=True)
        clear_comfyui_cache()

        return io.NodeOutput(te_cond)

class LongCat_Video_SM_Audio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_Audio",
            display_name="LongCat Avatar Audio Encode",
            category="LongCat Avatar",
            inputs=[
                io.AudioEncoder.Input("audio_encoder"),
                io.Audio.Input("audio"),
                io.Int.Input("save_fps", default=25, min=8, max=1024, step=1),
                io.Combo.Input("audio_type",options= ['para', 'add']),
                io.String.Input("p_box",default="",multiline=False),
                io.Audio.Input("left_audio",optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="au_cond"),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder,audio,save_fps,audio_type,p_box,left_audio=None,num_segments=None) -> io.NodeOutput:
        parsed_p_box = parse_person_boxes(p_box)
        runtime_device = get_runtime_device()

        # IMPORTANT: legacy exported workflows may still carry num_segments; duration is now audio-driven.
        au_cond=get_audio_emb(audio_encoder,audio,left_audio,audio_type,save_fps,runtime_device,p_box=parsed_p_box)
        clear_comfyui_cache()
        return io.NodeOutput(au_cond)


class LongCat_Video_SM_AudioWindow(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_AudioWindow",
            display_name="LongCat Avatar Audio Window",
            category="LongCat Avatar",
            inputs=[
                io.Conditioning.Input("au_cond"),
                io.Int.Input("frames_processed", default=0, min=0, max=100000, step=1),
                io.Int.Input("num_frames", default=93, min=1, max=256, step=1),
                io.Int.Input("overlap", default=13, min=0, max=32, step=1),
                io.Combo.Input("if_not_enough_audio", options=["clamp", "mirror_from_end"]),
                io.Int.Input("ref_img_index", default=10, min=0, max=1024, step=1),
                io.Int.Input("mask_frame_range", default=3, min=0, max=1024, step=1),
            ],
            outputs=[
                io.Conditioning.Output(display_name="au_cond_window"),
            ],
        )

    @classmethod
    def execute(
        cls,
        au_cond,
        frames_processed,
        num_frames,
        overlap,
        if_not_enough_audio,
        ref_img_index,
        mask_frame_range,
    ) -> io.NodeOutput:
        au_cond_window = build_audio_window_payload(
            au_cond,
            frames_processed=frames_processed,
            num_frames=num_frames,
            overlap=overlap,
            if_not_enough_audio=if_not_enough_audio,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
        )
        clear_comfyui_cache()
        return io.NodeOutput(au_cond_window)


class LongCat_Video_SM_AudioCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_AudioCrop",
            display_name="LongCat Avatar Audio Crop",
            category="LongCat Avatar",
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("start_time", default="0:00", multiline=False),
                io.String.Input("end_time", default="1:00", multiline=False),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, start_time="0:00", end_time="1:00") -> io.NodeOutput:
        cropped_audio = crop_audio_payload(audio, start_time=start_time, end_time=end_time)
        # IMPORTANT: keep preview generation on ComfyUI's native temp-audio path; downstream nodes receive AUDIO unchanged.
        return io.NodeOutput(cropped_audio, ui=ui.PreviewAudio(cropped_audio, cls=cls))

class LongCat_Video_SM_Vocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_Vocal",
            display_name="LongCat Avatar Vocal Extract",
            category="LongCat Avatar",
            inputs=[
                io.AudioEncoder.Input("audio_encoder"),
                io.Audio.Input("audio"),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="audio_path"),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder,audio,) -> io.NodeOutput:
        audio_path,audio=get_audio_vocal(audio_encoder,audio2path(audio),folder_paths.get_output_directory())
        return io.NodeOutput(audio,audio_path)

class LongCat_Video_SM_WhisperModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_WhisperModel",
            display_name="LongCat Avatar Whisper",
            category="LongCat Avatar",
            inputs=[
                io.Combo.Input(
                    "audio_encoder",options=folder_paths.get_filename_list("audio_encoders") ,
                ),
            ],
            outputs=[
                io.AudioEncoder.Output(),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder) -> io.NodeOutput:
        # IMPORTANT: Avatar 1.5 is Whisper-only; Wav2Vec2 can load permissively and break lip sync.
        validate_longcat_avatar_whisper_model_name(audio_encoder)
        a_checkpoint_path = folder_paths.get_full_path_or_raise("audio_encoders", audio_encoder)
        from .LongCat_Video.longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
        audio_encoder = get_audio_encoder(a_checkpoint_path, 'avatar-v1.5',os.path.join(node_longcat_path, "LongCat_Video/whisper-large-v3"))
        audio_feature_extractor = get_audio_feature_extractor(os.path.join(node_longcat_path, "LongCat_Video/whisper-large-v3"), 'avatar-v1.5')
        audio_encoder={"audio_encoder":audio_encoder,"audio_feature_extractor":audio_feature_extractor}
        return io.NodeOutput(audio_encoder)

class LongCat_Video_SM_VocalModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_VocalModel",
            display_name="LongCat Avatar Vocal Model",
            category="LongCat Avatar",
            inputs=[
                io.Combo.Input(
                    "audio_encoder_vocal",options=["none"]+[i for i in folder_paths.get_filename_list("longcat") if i.endswith(".onnx")],
                ),
            ],
            outputs=[
                io.AudioEncoder.Output(),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder_vocal) -> io.NodeOutput:
        vocal_separator_path=folder_paths.get_full_path_or_raise("longcat", audio_encoder_vocal) if audio_encoder_vocal!="none" else None
        audio_encoder=load_audio_vocal(vocal_separator_path,folder_paths.get_output_directory(),weigths_longcat_current_path)
        return io.NodeOutput(audio_encoder)
