import os
import json
import time
import random
import uuid
import shutil
import PIL.Image
import numpy as np
from pathlib import Path

import torch
#import torch.distributed as dist

from transformers import AutoTokenizer
from diffusers.utils import load_image

from .longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline,get_audio_embedding_whisper,get_audio_embedding_whisper_
from .longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from .longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from .longcat_video.modules.quantization import load_quantized_dit,get_config
# from .longcat_video.context_parallel import context_parallel_util
from .model_contract import (
    AVATAR_V15,
    normalize_sampling_parameters,
    resolve_avatar_model_contract,
    validate_state_dict_result,
)
from .model_loading_contract import resolve_dit_load_plan
from .attention_contract import (
    apply_attention_mode_to_config,
    attention_mode_config_overrides,
    print_attention_diagnostics,
    normalize_attention_mode,
    validate_attention_mode_availability,
)
from .audio_contract import (
    build_avatar_audio_payload,
    calculate_generate_duration,
    calculate_num_segments_for_prepared_audio_lengths,
    calculate_source_sample_count_for_prepared_audio_lengths,
    calculate_target_output_frames_for_sample_count,
    ensure_mono_waveform_array,
    resolve_audio_embedding_device,
    target_sample_count,
    validate_audio_conditioning_payload,
    validate_audio_embedding,
    validate_audio_type,
    validate_matching_audio_embedding_shapes,
    validate_multi_audio_lengths,
)
from .backend_dtype_policy import resolve_random_generator_device
from .sampler_contract import (
    expected_output_frames,
    normalize_seed,
    resolve_resolution_dimensions,
    segment_audio_start,
    trim_output_tensor_to_target_frames,
    validate_continuation_parameters,
    validate_output_tensor_contract,
)
from .debug_profile import ensure_debug_profiler

# -------- avatar related --------
import librosa
from .longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
# from .longcat_video.audio_process.torch_utils import save_video_ffmpeg
from safetensors.torch import load_file as safe_load


def _load_sharded_state_dict(shard_paths):
    state_dict = {}
    for shard_path in shard_paths:
        state_dict.update(safe_load(shard_path, device="cpu"))
    return state_dict

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _get_vocal_separator_class():
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise ImportError(
            "LongCat Avatar Vocal Extract requires vocal separation dependencies. "
            "Install them with `pip install -r requirements.txt` in the ComfyUI Python environment."
        ) from exc
    return Separator

def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    uid = timestamp_part + random_part
    return uid

def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separate failed. Using raw audio.")
        return None

    default_vocal_path = Path(os.path.join(audio_output_dir_temp ,"vocals", f"{outputs[0]}"))
    default_vocal_path = default_vocal_path.resolve().as_posix()
    # cmd = f"mv '{default_vocal_path}' '{target_path}'"
    # os.system(cmd)
    shutil.move(default_vocal_path, target_path)
    return target_path

def replace_to_vocal_suffix(raw_speech_path):
    """Replace the suffix of the raw speech path with _vocal.wav"""
    path = Path(raw_speech_path)
    new_name = path.stem + "_vocal" + path.suffix
    return str(path.with_name(new_name))

def load_longcat_video_model(model_path,vae_path,distill_checkpoint_path="", node_longcat_path="",
                             use_int8=False,model_type=AVATAR_V15,
                             checkpoint_source="single_file_safetensors",
                             official_checkpoint_path=None,
                             attention_mode="auto",):

    contract = resolve_avatar_model_contract(
        model_path,
        vae_path,
        distill_checkpoint_path,
        node_longcat_path,
        use_int8=bool(use_int8),
        model_type=model_type,
        checkpoint_source=checkpoint_source,
        official_checkpoint_path=official_checkpoint_path,
    )
    use_distill = contract.use_distill
    attention_mode = normalize_attention_mode(attention_mode)
    validate_attention_mode_availability(attention_mode)
    attention_overrides = attention_mode_config_overrides(attention_mode)

    # initialize models
    tokenizer = AutoTokenizer.from_pretrained(contract.metadata_root, subfolder=contract.tokenizer_subfolder, torch_dtype=torch.bfloat16)

    vae_config=AutoencoderKLWan.load_config(contract.vae_config_path)
    vae=AutoencoderKLWan.from_config(vae_config, torch_dtype=torch.bfloat16)
    vae_sd=safe_load(contract.vae_path, device="cpu")
    vae_result = vae.load_state_dict(vae_sd, strict=False)
    validate_state_dict_result("VAE", vae_result)
    vae=vae.eval().to(torch.bfloat16)
    del vae_sd
    #vae = AutoencoderKLWan.from_single_file(vae_path,config=os.path.join(node_longcat_path, 'LongCat_Video/LongCat-Video/vae/config.json'), torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(contract.metadata_root, subfolder=contract.scheduler_subfolder, torch_dtype=torch.bfloat16)

    load_plan = resolve_dit_load_plan(contract)
    if contract.model_format in ("safetensors", "sharded_safetensors"):
        if load_plan.loader_kind == "official_sharded_int8":
            print("[INFO] Loading official sharded INT8 quantized DiT model...")
            dit = load_quantized_dit(
                load_plan.checkpoint_root,
                subfolder=load_plan.subfolder,
                cp_split_hw=[1,1],
                single_file=None,
                attention_mode=attention_mode,
                **attention_overrides,
            )
        elif load_plan.loader_kind == "single_file_int8":
            print("[INFO] Loading INT8 quantized DiT model...")
            dit = load_quantized_dit(
                contract.metadata_root,
                subfolder="base_model_int8",
                cp_split_hw=[1,1],
                single_file=contract.model_path,
                attention_mode=attention_mode,
                **attention_overrides,
            )
        elif load_plan.loader_kind in ("single_file_safetensors", "official_sharded"):
            print("[INFO] Loading normal  DiT model...")
            config_path = (
                os.path.join(load_plan.checkpoint_root, load_plan.subfolder)
                if load_plan.loader_kind == "official_sharded"
                else contract.dit_config_dir
            )
            #config=LongCatVideoAvatarTransformer3DModel.load_config(config_path)
            config=get_config(config_path,cp_split_hw=[1,1])
            config=apply_attention_mode_to_config(config, attention_mode)
            print_attention_diagnostics(attention_mode, config)
            with torch.device('meta'):
                # Diffusers ConfigMixin requires the config dict as the first positional argument.
                dit=LongCatVideoAvatarTransformer3DModel.from_config(config)
            sd=(
                _load_sharded_state_dict(load_plan.shard_paths)
                if load_plan.loader_kind == "official_sharded"
                else safe_load(contract.model_path, device="cpu")
            )
            X=dit.load_state_dict(sd, strict=False,assign=True)
            validate_state_dict_result("DiT", X)
            del sd
            dit=dit.eval().to(torch.bfloat16)
            #dit.cp_split_hw=[1,1]
            #dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(os.path.join(node_longcat_path, 'LongCat_Video/LongCat-Video'), subfolder="base_model", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unsupported DiT load plan: {load_plan.loader_kind}")
    else:
        raise ValueError(f"Unsupported model format: {contract.model_format}")
    if use_distill:
        dit.load_lora(contract.distill_checkpoint_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
        dit.enable_loras(["dmd"])
    # initialize pipeline
    pipe = LongCatVideoAvatarPipeline(
        tokenizer = tokenizer,
        text_encoder = None,
        vae = vae,
        scheduler = scheduler,
        dit = dit,
        audio_encoder=None,
        audio_feature_extractor=None,
        model_type=contract.model_type
    )
    #pipe.to(local_rank)
    pipe.use_distill = use_distill
    pipe.use_int8 = contract.use_int8
    pipe.model_type = contract.model_type
    pipe.model_contract = contract
    pipe.effective_num_inference_steps = contract.effective_num_inference_steps
    pipe.effective_text_guidance_scale = contract.effective_text_guidance_scale
    pipe.effective_audio_guidance_scale = contract.effective_audio_guidance_scale
    pipe.scheduler_source = contract.scheduler_source
    return pipe

def audio_prepare_multi(left_speech_array, right_speech_array, generate_duration,  sr=16000, audio_type='para'):
    validate_audio_type(audio_type)
    #left_speech_array, right_speech_array = None, None
    # if left_temp_vocal_path is not None:
    #     left_speech_array, sr = librosa.load(left_temp_vocal_path, sr=sample_rate)
    #     left_raw_speech_array, _ = librosa.load(left_raw_speech_path, sr=sample_rate)

    # if right_temp_vocal_path is not None:
    #     right_speech_array, sr = librosa.load(right_temp_vocal_path, sr=sample_rate)
    #     right_raw_speech_array, _ = librosa.load(right_raw_speech_path, sr=sample_rate)

    if left_speech_array is None:
        left_speech_array = np.zeros_like(right_speech_array)
        #left_raw_speech_array = np.zeros_like(right_raw_speech_array)

    if right_speech_array is None:
        right_speech_array = np.zeros_like(left_speech_array)
        #right_raw_speech_array = np.zeros_like(left_raw_speech_array)

    if audio_type == 'add':
        left_speech_array_ext = np.concatenate([left_speech_array, np.zeros_like(right_speech_array)])
        right_speech_array_ext = np.concatenate([np.zeros_like(left_speech_array), right_speech_array])
        #merge_raw_speech = np.concatenate([left_raw_speech_array, np.zeros_like(right_raw_speech_array)]) + np.concatenate([np.zeros_like(left_raw_speech_array), right_raw_speech_array])
    elif audio_type == 'para':
        left_speech_array_ext = left_speech_array
        right_speech_array_ext = right_speech_array
        #merge_raw_speech = left_raw_speech_array + right_raw_speech_array
    else:
        raise NotImplementedError(f"Unsupported audio_type of {audio_type}")

    validate_multi_audio_lengths(len(left_speech_array_ext), len(right_speech_array_ext), 'para')


    added_sample_nums = target_sample_count(generate_duration, sr) - len(left_speech_array_ext)
    if added_sample_nums > 0:
        left_speech_array_ext  = np.append(left_speech_array_ext, [0.]*added_sample_nums)
        right_speech_array_ext = np.append(right_speech_array_ext, [0.]*added_sample_nums)

    return left_speech_array_ext, right_speech_array_ext

def prepare_audio(audio, sample_rate=16000):
    speech_array = np.array(audio["waveform"].squeeze(0), dtype=np.float32)
    speech_array = ensure_mono_waveform_array(speech_array)
    speech_array = np.asarray(speech_array, dtype=np.float32)
    sr = audio["sample_rate"]
    if sr != 16000:
        import librosa
        speech_array = librosa.resample(speech_array, orig_sr=sr, target_sr=sample_rate)
        sr = 16000
    return speech_array,sr

def get_audio_emb(audio_encoder,audio,left_audio,audio_type,save_fps,device,p_box,model_type='avatar-v1.5' ):
    num_frames=93
    num_cond_frames = 13
    audio_stride=1
    audio_embedding_device = resolve_audio_embedding_device(device)
    validate_audio_type(audio_type)

    speech_array, sr = prepare_audio(audio)
    if left_audio is not None:
        left_speech_array, _ = prepare_audio(left_audio)
    else:
        left_speech_array = None
    left_person_bbox, right_person_bbox,back_full_audio_emb,left_full_audio_emb,other_person_bbox = None, None, None, None,None
    use_background_silent_audio = False
    if p_box is not None:
        # bbox: [left_y_min, left_x_min, left_y_max, left_x_max]
        # x and y coordinates correspond to the width and height dimensions, respectively
        left_person_bbox=p_box[0]
        right_person_bbox=p_box[1]
        other_person_bbox = p_box[2] if len(p_box) > 2 else None
        use_background_silent_audio = other_person_bbox is not None and len(other_person_bbox) > 0

    if left_speech_array is not None:
        source_sample_count = calculate_source_sample_count_for_prepared_audio_lengths(
            len(speech_array),
            audio_type=audio_type,
            left_sample_count=len(left_speech_array),
        )
        target_output_frames = calculate_target_output_frames_for_sample_count(source_sample_count, sr, save_fps)
        num_segments = calculate_num_segments_for_prepared_audio_lengths(
            len(speech_array),
            sr,
            save_fps,
            audio_type=audio_type,
            left_sample_count=len(left_speech_array),
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
        )
        generate_duration = calculate_generate_duration(
            save_fps,
            num_segments,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
        )
        left_speech_array_ext, right_speech_array_ext = audio_prepare_multi(left_speech_array,speech_array, generate_duration, sr=sr, audio_type=audio_type)
        if isinstance(audio_encoder,dict):
            left_full_audio_emb = get_audio_embedding_whisper(audio_encoder["audio_encoder"].to(audio_embedding_device), audio_encoder["audio_feature_extractor"], left_speech_array_ext, fps=save_fps*audio_stride, device=audio_embedding_device, sample_rate=sr)
            full_audio_emb = get_audio_embedding_whisper(audio_encoder["audio_encoder"].to(audio_embedding_device), audio_encoder["audio_feature_extractor"], right_speech_array_ext, fps=save_fps*audio_stride, device=audio_embedding_device, sample_rate=sr)
        else:
            left_full_audio_emb = get_audio_embedding_whisper_(audio_encoder, left_speech_array_ext, fps=save_fps*audio_stride, )
            full_audio_emb = get_audio_embedding_whisper_(audio_encoder, right_speech_array_ext, fps=save_fps*audio_stride,)

        validate_matching_audio_embedding_shapes(
            ("left_full_audio_emb", left_full_audio_emb),
            ("full_audio_emb", full_audio_emb),
        )
        if use_background_silent_audio:
            if isinstance(audio_encoder,dict):
                back_full_audio_emb = get_audio_embedding_whisper(audio_encoder["audio_encoder"].to(audio_embedding_device), audio_encoder["audio_feature_extractor"], np.zeros_like(left_speech_array_ext), fps=save_fps*audio_stride, device=audio_embedding_device, sample_rate=sr)
            else:
                back_full_audio_emb = get_audio_embedding_whisper_(audio_encoder,np.zeros_like(left_speech_array_ext), fps=save_fps*audio_stride, )
        if use_background_silent_audio:
            validate_matching_audio_embedding_shapes(
                ("left_full_audio_emb", left_full_audio_emb),
                ("back_full_audio_emb", back_full_audio_emb),
            )
    else:
        target_output_frames = calculate_target_output_frames_for_sample_count(len(speech_array), sr, save_fps)
        num_segments = calculate_num_segments_for_prepared_audio_lengths(
            len(speech_array),
            sr,
            save_fps,
            audio_type=audio_type,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
        )
        generate_duration = calculate_generate_duration(
            save_fps,
            num_segments,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
        )
        added_sample_nums = target_sample_count(generate_duration, sr) - len(speech_array)
        if added_sample_nums > 0:
            speech_array = np.append(speech_array, [0.]*added_sample_nums)
        if isinstance(audio_encoder,dict):
            full_audio_emb = get_audio_embedding_whisper(audio_encoder["audio_encoder"].to(audio_embedding_device), audio_encoder["audio_feature_extractor"], speech_array, fps=save_fps*audio_stride, device=audio_embedding_device, sample_rate=sr)
        else:
            full_audio_emb=get_audio_embedding_whisper_(audio_encoder, speech_array, fps=save_fps*audio_stride, ) #torch.Size([2142, 5, 1280])
    validate_audio_embedding(full_audio_emb, "full_audio_emb")
    if isinstance(audio_encoder,dict):
        audio_encoder["audio_encoder"].to("cpu")
        del audio_encoder
    return build_avatar_audio_payload(
        full_audio_emb=full_audio_emb,
        left_full_audio_emb=left_full_audio_emb,
        back_full_audio_emb=back_full_audio_emb,
        num_segments=num_segments,
        audio_stride=audio_stride,
        save_fps=save_fps,
        audio_type=audio_type,
        left_person_bbox=left_person_bbox,
        right_person_bbox=right_person_bbox,
        other_person_bbox=other_person_bbox,
        use_background_silent_audio=use_background_silent_audio,
        target_output_frames=target_output_frames,
    )


def load_audio_vocal(vocal_separator_path,audio_output_dir_temp,checkpoint_dir):
    if vocal_separator_path is None:
        vocal_separator_path = os.path.join(checkpoint_dir, 'Kim_Vocal_2.onnx')
    os.makedirs(audio_output_dir_temp, exist_ok=True)
    audio_output_dir_temp = Path(audio_output_dir_temp)
    audio_separator_model_path = os.path.dirname(vocal_separator_path)
    audio_separator_model_name = os.path.basename(vocal_separator_path)
    Separator = _get_vocal_separator_class()
    vocal_separator = Separator(
        output_dir=audio_output_dir_temp / "vocals",
        output_single_stem="vocals",
        model_file_dir=audio_separator_model_path,
    )

    vocal_separator.load_model(audio_separator_model_name)
    vocal_separator.onnx_execution_provider = ["CUDAExecutionProvider"]
    return vocal_separator


def get_audio_vocal(vocal_separator,raw_speech_path,audio_output_dir_temp,):

    vocal_path=replace_to_vocal_suffix(raw_speech_path)
    os.makedirs(os.path.dirname(vocal_path), exist_ok=True)

    temp_vocal_path = extract_vocal_from_speech(raw_speech_path,vocal_path , vocal_separator, audio_output_dir_temp)
    import librosa
    vocal_array, sr = librosa.load(temp_vocal_path, sr=16000)
    #print("vocal_array.shape", vocal_array.shape)
    audio={
        "waveform": torch.from_numpy(vocal_array).unsqueeze(0).unsqueeze(0),
        "sample_rate": sr,
    }
    return temp_vocal_path,audio

def generate(pipe,condition,te_cond,device,seed,stage_1,cond_image,resolution,
             text_guidance_scale,audio_guidance_scale,num_inference_steps,ref_img_index,mask_frame_range,
             use_distill,model_type=AVATAR_V15,debug_profile=None):
    debug_profile = ensure_debug_profiler(debug_profile)
    num_frames=93 # 滑动窗口13,硬编码为93确保滑动窗口覆盖整个视频
    num_cond_frames=13
    audio_stride=condition['audio_stride']
    full_audio_emb=condition['full_audio_emb']
    num_segments=condition['num_segments']

    # prepare audio embedding for the first clip
    with debug_profile.phase("segment_1.audio_slice", num_frames=num_frames, audio_stride=audio_stride):
        indices = torch.arange(2 * 2 + 1) - 2
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0]-1)
        audio_emb = full_audio_emb[center_indices][None,...].to(device)

    model_type = getattr(pipe, "model_type", model_type)
    num_inference_steps, text_guidance_scale, audio_guidance_scale = normalize_sampling_parameters(
        model_type,
        use_distill,
        num_inference_steps,
        text_guidance_scale,
        audio_guidance_scale,
    )

    height, width = resolve_resolution_dimensions(resolution)
    seed = normalize_seed(seed)
    ref_img_index, mask_frame_range = validate_continuation_parameters(ref_img_index, mask_frame_range)
    generator = torch.Generator(device=resolve_random_generator_device(device))
    generator.manual_seed(seed)


    #if local_rank == 0:
    print(f"Generating {stage_1} 1/{num_segments}...")

    if stage_1 == 'at2v':
        # ==============================
        #          at2v (480P)
        # ==============================
        output_tuple = pipe.generate_at2v(
            prompt=None,
            negative_prompt=None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            audio_emb=audio_emb,
            use_distill=use_distill,
            prompt_embeds=te_cond["prompt_embeds"],
            negative_prompt_embeds=te_cond["negative_prompt_embeds"],
            text=te_cond["text"],
            debug_profile=debug_profile.child("segment_1.at2v"),
        )
        output, latent = output_tuple
        with debug_profile.phase("segment_1.output_to_cpu_frames"):
            output = output[0]
            video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
            video = [PIL.Image.fromarray(img) for img in video]

            #if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(video)).float() / 255.0
            validate_output_tensor_contract(output_tensor)
        #save_video_ffmpeg(output_tensor, os.path.join(output_dir, "at2v_demo_1"), raw_speech_path, fps=save_fps, quality=5)
        del output
        with debug_profile.phase("segment_1.torch_gc"):
            torch_gc()

    elif stage_1 == 'ai2v':
        # ==============================
        #          ai2v (480P)
        # ==============================
        image_path =cond_image # input_data['cond_image']
        image = load_image(image_path)
        output_tuple = pipe.generate_ai2v(
            image=image,
            prompt=None,
            negative_prompt=None,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            output_type='both',
            generator=generator,
            audio_emb=audio_emb,
            use_distill=use_distill,
            prompt_embeds=te_cond["prompt_embeds"],
            negative_prompt_embeds=te_cond["negative_prompt_embeds"],
            text=te_cond["text"],
            debug_profile=debug_profile.child("segment_1.ai2v"),
        )
        output, latent = output_tuple
        with debug_profile.phase("segment_1.output_to_cpu_frames"):
            output = output[0]
            video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
            video = [PIL.Image.fromarray(img) for img in video]

            #if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(video)).float() / 255.0
            validate_output_tensor_contract(output_tensor)
        #save_video_ffmpeg(output_tensor, os.path.join(output_dir, "ai2v_demo_1"), raw_speech_path, fps=save_fps, quality=5)
        del output
        with debug_profile.phase("segment_1.torch_gc"):
            torch_gc()
    else:
        raise NotImplementedError(f"Not supported type of stage_1: {stage_1}")

    # if context_parallel_util.get_cp_size() > 1:
    #     torch.distributed.barrier(group=context_parallel_util.get_cp_group())

    # =========================================
    #         long video generation (480P)
    # =========================================
    # load parsed long video args
    #ref_img_index = args.ref_img_index
    #mask_frame_range = args.mask_frame_range

    width, height = video[0].size
    current_video = video
    ref_latent = latent[:, :, :1].clone()
    all_generated_frames = video

    for segment_idx in range(1, num_segments):
        #if local_rank == 0:
        print(f"Generating segment {segment_idx+1}/{num_segments}...")

        # prepare audio embedding for the next clip
        segment_label = f"segment_{segment_idx + 1}"
        with debug_profile.phase(f"{segment_label}.audio_slice", audio_stride=audio_stride):
            audio_start_idx = segment_audio_start(segment_idx, audio_stride)
            audio_end_idx   = audio_start_idx + audio_stride * num_frames
            center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0]-1)
            audio_emb = full_audio_emb[center_indices][None,...].to(device)

        output_tuple = pipe.generate_avc(
            video=current_video,
            video_latent=latent,
            prompt=None,
            negative_prompt=None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            use_kv_cache=True,
            # CRITICAL: official Avatar continuation keeps KV cache on GPU; CPU offload causes per-step host/device transfers and segment slowdown.
            offload_kv_cache=False,
            enhance_hf=True if not use_distill else False,
            audio_emb=audio_emb,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            use_distill=use_distill,
            prompt_embeds=te_cond["prompt_embeds"],
            negative_prompt_embeds=te_cond["negative_prompt_embeds"],
            text=te_cond["text"],
            debug_profile=debug_profile.child(f"{segment_label}.avc"),
        )
        output, latent = output_tuple

        with debug_profile.phase(f"{segment_label}.output_to_cpu_frames"):
            output = output[0]
            new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
            new_video = [PIL.Image.fromarray(img) for img in new_video]
            del output

            all_generated_frames.extend(new_video[num_cond_frames:])

            current_video = new_video

            #if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(all_generated_frames)/255.0).float()
            validate_output_tensor_contract(output_tensor, expected_frames=expected_output_frames(segment_idx + 1))
        # save_video_ffmpeg(output_tensor, os.path.join(output_dir, f"video_continue_{segment_idx+1}"), raw_speech_path, fps=save_fps, quality=5)
        # del output_tensor
    with debug_profile.phase("final_output_contract"):
        validate_output_tensor_contract(output_tensor, expected_frames=expected_output_frames(num_segments))
    # CRITICAL: generated segments include padded silent coverage; return only source-audio frames for lip-sync.
    with debug_profile.phase("trim_to_target_frames"):
        output_tensor = trim_output_tensor_to_target_frames(output_tensor, condition.get("target_output_frames"))
    return output_tensor
