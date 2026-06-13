import os
import json
import time
import math
import random
import argparse
import datetime
import shutil
import PIL.Image
import numpy as np
from pathlib import Path

import torch
#import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.utils import load_image

from .longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from .longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from .longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from .longcat_video.modules.quantization import load_quantized_dit
# from .longcat_video.context_parallel import context_parallel_util

# -------- avatar related --------
import librosa
import soundfile as sf
from .longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from .longcat_video.audio_process.torch_utils import save_video_ffmpeg
from .model_contract import AVATAR_V15, normalize_sampling_parameters
from .sampler_contract import (
    expected_output_frames,
    normalize_seed,
    resolve_resolution_dimensions,
    segment_audio_start,
    trim_output_tensor_to_target_frames,
    validate_continuation_parameters,
    validate_output_tensor_contract,
)
from .backend_dtype_policy import resolve_random_generator_device
from .debug_profile import ensure_debug_profiler


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    uid = timestamp_part + random_part
    return uid

def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    if source_path is None:
        return None

    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separate failed. Using raw audio.")
        return None

    default_vocal_path = audio_output_dir_temp / "vocals" / outputs[0]
    default_vocal_path = default_vocal_path.resolve().as_posix()
    shutil.move(default_vocal_path, target_path)
    return target_path

def audio_prepare_multi(left_temp_vocal_path, right_temp_vocal_path, generate_duration, left_raw_speech_path, right_raw_speech_path, sample_rate=16000, audio_type='para'):
    left_speech_array, right_speech_array = None, None
    if left_temp_vocal_path is not None:
        left_speech_array, sr = librosa.load(left_temp_vocal_path, sr=sample_rate)
        left_raw_speech_array, _ = librosa.load(left_raw_speech_path, sr=sample_rate)

    if right_temp_vocal_path is not None:
        right_speech_array, sr = librosa.load(right_temp_vocal_path, sr=sample_rate)
        right_raw_speech_array, _ = librosa.load(right_raw_speech_path, sr=sample_rate)

    if left_speech_array is None:
        left_speech_array = np.zeros_like(right_speech_array)
        left_raw_speech_array = np.zeros_like(right_raw_speech_array)

    if right_speech_array is None:
        right_speech_array = np.zeros_like(left_speech_array)
        right_raw_speech_array = np.zeros_like(left_raw_speech_array)

    if audio_type == 'add':
        left_speech_array_ext = np.concatenate([left_speech_array, np.zeros_like(right_speech_array)])
        right_speech_array_ext = np.concatenate([np.zeros_like(left_speech_array), right_speech_array])
        merge_raw_speech = np.concatenate([left_raw_speech_array, np.zeros_like(right_raw_speech_array)]) + \
                            np.concatenate([np.zeros_like(left_raw_speech_array), right_raw_speech_array])
    elif audio_type == 'para':
        left_speech_array_ext = left_speech_array
        right_speech_array_ext = right_speech_array
        if len(left_speech_array_ext) != len(right_speech_array_ext):
            raise ValueError("audio_type='para' requires equal-length left and right audio.")
        merge_raw_speech = left_raw_speech_array + right_raw_speech_array
    else:
        raise NotImplementedError(f"Unsupported audio_type of {audio_type}")


    source_duraion = len(left_speech_array_ext) / sr
    added_sample_nums = math.ceil((generate_duration - source_duraion) * sr)
    if added_sample_nums > 0:
        left_speech_array_ext  = np.append(left_speech_array_ext, [0.]*added_sample_nums)
        right_speech_array_ext = np.append(right_speech_array_ext, [0.]*added_sample_nums)

    return left_speech_array_ext, right_speech_array_ext, merge_raw_speech


def resolve_person_bbox_coordinates(src_width, src_height, left_person_bbox, right_person_bbox):
    if left_person_bbox is None and right_person_bbox is None:
        face_scale = 0.1
        left_y_min, left_y_max = int(src_height * face_scale), int(src_height * (1 - face_scale))
        right_y_min, right_y_max = left_y_min, left_y_max
        half_width = src_width // 2
        left_x_min, left_x_max = int(half_width * face_scale), int(half_width * (1 - face_scale))
        right_x_min = int(half_width * face_scale + half_width)
        right_x_max = int(half_width * (1 - face_scale) + half_width)
    elif left_person_bbox is not None and right_person_bbox is not None:
        left_y_min, left_x_min, left_y_max, left_x_max = left_person_bbox
        right_y_min, right_x_min, right_y_max, right_x_max = right_person_bbox
    else:
        raise ValueError("Multi-audio generation requires both left and right person boxes, or neither.")
    return left_y_min, left_x_min, left_y_max, left_x_max, right_y_min, right_x_min, right_y_max, right_x_max


def generate_multi(pipe,condition,te_cond,device,seed,cond_image,resolution,
             text_guidance_scale,audio_guidance_scale,num_inference_steps,ref_img_index,mask_frame_range,use_distill,debug_profile=None):
    debug_profile = ensure_debug_profiler(debug_profile)

    # load parsed args
    # input_json = args.input_json
    # checkpoint_dir = args.checkpoint_dir
    # context_parallel_size = args.context_parallel_size
    # num_inference_steps = args.num_inference_steps
    # text_guidance_scale = args.text_guidance_scale
    # audio_guidance_scale = args.audio_guidance_scale
    # resolution = args.resolution
    # num_segments = max(1, args.num_segments)
    # output_dir = args.output_dir
    # model_type = args.model_type
    # use_distill = args.use_distill
    # use_int8 = args.use_int8

    # if use_distill and model_type == "avatar-v1.5":
    #     num_inference_steps = 8
    #     text_guidance_scale = 1.0
    #     audio_guidance_scale = 1.0

    # # set up default inference params
    # save_fps = 16
    # audio_stride = 2
    # if model_type == "avatar-v1.5":
    #     save_fps = 25
    #     audio_stride = 1
    # num_frames = 93
    # num_cond_frames = 13

    # if resolution == '480p':
    #     height, width = 480, 832
    # elif resolution == '720p':
    #     height, width = 768, 1280

    # # case setup
    # with open(input_json, 'r', encoding='utf-8') as f:
    #     input_data = json.load(f)
    # prompt = input_data['prompt']
    # negative_prompt = "Close-up, bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    # left_raw_speech_path = input_data['cond_audio'].get('person1', None)
    # right_raw_speech_path = input_data['cond_audio'].get('person2', None)
    # left_person_bbox, right_person_bbox = None, None

    num_frames=93 # 滑动窗口13,硬编码为93确保滑动窗口覆盖整个视频
    num_cond_frames=13
    audio_stride=condition['audio_stride']
    right_full_audio_emb=condition['full_audio_emb']
    left_full_audio_emb=condition['left_full_audio_emb']
    back_full_audio_emb=condition['back_full_audio_emb']
    num_segments=condition['num_segments']
    use_background_silent_audio = condition['use_background_silent_audio']
    left_person_bbox = condition['left_person_bbox']
    right_person_bbox = condition['right_person_bbox']
    other_person_bbox = condition['other_person_bbox']
    resolve_resolution_dimensions(resolution)
    seed = normalize_seed(seed)
    ref_img_index, mask_frame_range = validate_continuation_parameters(ref_img_index, mask_frame_range)
    model_type = getattr(pipe, "model_type", AVATAR_V15)
    num_inference_steps, text_guidance_scale, audio_guidance_scale = normalize_sampling_parameters(
        model_type,
        use_distill,
        num_inference_steps,
        text_guidance_scale,
        audio_guidance_scale,
    )
    # if 'bbox' in input_data:
    #     # bbox: [left_y_min, left_x_min, left_y_max, left_x_max]
    #     # x and y coordinates correspond to the width and height dimensions, respectively
    #     left_person_bbox = input_data['bbox'].get('person1', None)
    #     right_person_bbox = input_data['bbox'].get('person2', None)
    #     other_person_bbox = input_data['bbox'].get('others', None)
    #     use_background_silent_audio = other_person_bbox is not None and len(other_person_bbox) > 0
    # audio_type = 'para'
    # if 'audio_type' in input_data:
    #     audio_type = input_data.get('audio_type', 'para')

    # prepare distributed environment
    # rank = int(os.environ['RANK'])
    # num_gpus = torch.cuda.device_count()
    # local_rank = rank % num_gpus
    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
    # global_rank    = dist.get_rank()
    # num_processes  = dist.get_world_size()

    # initialize context parallel
    # context_parallel_util.init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    # cp_rank = context_parallel_util.get_cp_rank()
    # cp_size = context_parallel_util.get_cp_size()
    # cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    # initialize models
    # tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="tokenizer", torch_dtype=torch.bfloat16)
    # text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="text_encoder", torch_dtype=torch.bfloat16)
    # vae = AutoencoderKLWan.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="vae", torch_dtype=torch.bfloat16)
    # if model_type == "avatar-v1.0":
    #     scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="scheduler", torch_dtype=torch.bfloat16)
    # elif model_type == "avatar-v1.5":
    #     scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    # else:
    #     raise ValueError(f"Unsupported model_type: {model_type}. Expected 'avatar-v1.0' or 'avatar-v1.5'.")

    # if model_type == "avatar-v1.0":
    #     dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="avatar_multi", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
    # elif model_type == "avatar-v1.5":
    #     if use_int8:
    #         print("[INFO] Loading INT8 quantized DiT model...")
    #         dit = load_quantized_dit(checkpoint_dir, subfolder="base_model_int8", cp_split_hw=cp_split_hw)
    #     else:
    #         dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="base_model", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
    #     if use_distill:
    #         distill_checkpoint_path = os.path.join(checkpoint_dir, 'lora', f'dmd_lora.safetensors')
    #         if os.path.exists(distill_checkpoint_path):
    #             dit.load_lora(distill_checkpoint_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
    #             dit.enable_loras(["dmd"])

    # initialize audio models
    # if model_type == "avatar-v1.0":
    #     audio_model_checkpoint_path = os.path.join(checkpoint_dir, 'chinese-wav2vec2-base')
    # elif model_type == "avatar-v1.5":
    #     audio_model_checkpoint_path = os.path.join(checkpoint_dir, 'whisper-large-v3')
    # audio_encoder = get_audio_encoder(audio_model_checkpoint_path, model_type).to(local_rank)
    # audio_feature_extractor = get_audio_feature_extractor(audio_model_checkpoint_path, model_type)

    # vocal_separator_path = os.path.join(checkpoint_dir, 'vocal_separator/Kim_Vocal_2.onnx')
    # audio_output_dir_temp = f"./audio_temp_file"
    # os.makedirs(audio_output_dir_temp, exist_ok=True)
    # audio_output_dir_temp = Path(audio_output_dir_temp)
    # audio_separator_model_path = os.path.dirname(vocal_separator_path)
    # audio_separator_model_name = os.path.basename(vocal_separator_path)
    # vocal_separator = Separator(
    #     output_dir=audio_output_dir_temp / "vocals",
    #     output_single_stem="vocals",
    #     model_file_dir=audio_separator_model_path,
    # )
    # vocal_separator.load_model(audio_separator_model_name)


    # initialize pipeline
    # pipe = LongCatVideoAvatarPipeline(
    #     tokenizer = tokenizer,
    #     text_encoder = text_encoder,
    #     vae = vae,
    #     scheduler = scheduler,
    #     dit = dit,
    #     audio_encoder=audio_encoder,
    #     audio_feature_extractor=audio_feature_extractor,
    #     model_type=model_type
    # )
    # pipe.to(local_rank)

    # global_seed = 42
    # seed = global_seed + global_rank

    generator = torch.Generator(device=resolve_random_generator_device(device))
    generator.manual_seed(seed)

    #if cp_rank == 0:
    # extract vocal
    # sr = 16000
    # left_temp_vocal_path  = os.path.join(audio_output_dir_temp, f"{generate_random_uid()}_left_temp_vocal.wav")
    # right_temp_vocal_path = os.path.join(audio_output_dir_temp, f"{generate_random_uid()}_right_temp_vocal.wav")
    # left_temp_vocal_path = extract_vocal_from_speech(left_raw_speech_path, left_temp_vocal_path, vocal_separator, audio_output_dir_temp)
    # right_temp_vocal_path = extract_vocal_from_speech(right_raw_speech_path, right_temp_vocal_path, vocal_separator, audio_output_dir_temp)

    # # prepare each vocal and synthesize the sum audio
    # generate_duration = num_frames / save_fps + (num_segments-1) * (num_frames-num_cond_frames) / save_fps
    # left_speech_array_ext, right_speech_array_ext, merge_speech = audio_prepare_multi(left_temp_vocal_path, right_temp_vocal_path, generate_duration, \
    #                                                                                     left_raw_speech_path, right_raw_speech_path, sample_rate=sr, audio_type=audio_type)
    # merge_speech_path = f"/tmp/temp_speech_{generate_random_uid()}_{global_rank}_merge.wav"
    # sf.write(merge_speech_path, merge_speech, 16000)

    # left_full_audio_emb = pipe.get_audio_embedding(left_speech_array_ext, fps=save_fps*audio_stride, device=local_rank, sample_rate=sr, model_type=model_type)
    # right_full_audio_emb = pipe.get_audio_embedding(right_speech_array_ext, fps=save_fps*audio_stride, device=local_rank, sample_rate=sr, model_type=model_type)
    # if use_background_silent_audio:
    #     back_full_audio_emb = pipe.get_audio_embedding(np.zeros_like(left_speech_array_ext), fps=save_fps*audio_stride, device=local_rank, sample_rate=sr, model_type=model_type)

    # if context_parallel_util.get_cp_size() > 1:
    #     full_audio_emb_shape_list = list(left_full_audio_emb.size())
    #     full_audio_emb_tensor_shape_list = torch.tensor(full_audio_emb_shape_list, dtype=torch.int64, device=left_full_audio_emb.device)
    #     context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
    #     context_parallel_util.cp_broadcast(left_full_audio_emb)
    #     context_parallel_util.cp_broadcast(right_full_audio_emb)
    #     if use_background_silent_audio:
    #         context_parallel_util.cp_broadcast(back_full_audio_emb)

    # if left_temp_vocal_path is not None and os.path.exists(left_temp_vocal_path):
    #     os.remove(left_temp_vocal_path)
    # if right_temp_vocal_path is not None and os.path.exists(right_temp_vocal_path):
    #     os.remove(right_temp_vocal_path)

    # elif context_parallel_util.get_cp_size() > 1:
    #     full_audio_emb_tensor_shape_list = torch.zeros(3, dtype=torch.int64, device=local_rank)
    #     context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
    #     full_audio_emb_shape_list = full_audio_emb_tensor_shape_list.tolist()
    #     left_full_audio_emb = torch.zeros(*full_audio_emb_shape_list, dtype=torch.float32, device=local_rank)
    #     context_parallel_util.cp_broadcast(left_full_audio_emb)
    #     right_full_audio_emb = torch.zeros(*full_audio_emb_shape_list, dtype=torch.float32, device=local_rank)
    #     context_parallel_util.cp_broadcast(right_full_audio_emb)
    #     if use_background_silent_audio:
    #         back_full_audio_emb = torch.zeros(*full_audio_emb_shape_list, dtype=torch.float32, device=local_rank)
    #         context_parallel_util.cp_broadcast(back_full_audio_emb)


    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames

    # get audio embedding for the first clip
    with debug_profile.phase("segment_1.audio_slice", audio_stride=audio_stride):
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=left_full_audio_emb.shape[0]-1)
        left_audio_emb = left_full_audio_emb[center_indices][None,...].to(device)
        right_audio_emb = right_full_audio_emb[center_indices][None,...].to(device)
        audio_embs = [left_audio_emb, right_audio_emb]
        if use_background_silent_audio:
            audio_embs.append(back_full_audio_emb[center_indices][None,...].to(device))
        audio_embs = torch.cat(audio_embs)


    # ==============================
    #          ai2v (480P)
    # ==============================
    #image_path = input_data['cond_image']
    image = load_image(cond_image)
    (src_width, src_height) = image.size

    # define human / background mask
    background_mask = torch.zeros([src_height, src_width])
    human_mask1 = torch.zeros([src_height, src_width])
    human_mask2 = torch.zeros([src_height, src_width])
    (
        left_y_min,
        left_x_min,
        left_y_max,
        left_x_max,
        right_y_min,
        right_x_min,
        right_y_max,
        right_x_max,
    ) = resolve_person_bbox_coordinates(src_width, src_height, left_person_bbox, right_person_bbox)
    human_mask1[left_y_min:left_y_max, left_x_min:left_x_max] = 1
    human_mask2[right_y_min:right_y_max, right_x_min:right_x_max] = 1
    background_mask += human_mask1
    background_mask += human_mask2
    background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
    total_mask = [human_mask1, human_mask2, background_mask]
    if use_background_silent_audio:
        for i in range(len(other_person_bbox)//4):
            other_person_bbox_i = other_person_bbox[i*4:(i+1)*4]
            other_person_mask = torch.zeros([src_height, src_width])
            other_person_mask[other_person_bbox_i[0]:other_person_bbox_i[2], other_person_bbox_i[1]:other_person_bbox_i[3]] = 1
            total_mask.append(other_person_mask)
    ref_target_masks = torch.stack(total_mask, dim=0).to(device)

    # generate video
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
        audio_emb=audio_embs,
        ref_target_masks=ref_target_masks,
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
    #save_video_ffmpeg(output_tensor, os.path.join(output_dir, "ai2v_demo_1"), merge_speech_path, fps=save_fps, quality=5)
    del output
    with debug_profile.phase("segment_1.torch_gc"):
        torch_gc()

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
            center_indices = torch.clamp(center_indices, min=0, max=left_full_audio_emb.shape[0]-1)
            left_audio_emb = left_full_audio_emb[center_indices][None,...].to(device)
            right_audio_emb = right_full_audio_emb[center_indices][None,...].to(device)
            audio_embs = [left_audio_emb, right_audio_emb]
            if use_background_silent_audio:
                audio_embs.append(back_full_audio_emb[center_indices][None,...].to(device))
            audio_embs = torch.cat(audio_embs)

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
            audio_emb=audio_embs,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            ref_target_masks=ref_target_masks,
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
            output_tensor =torch.from_numpy(np.array(all_generated_frames)/255.0).float()
            validate_output_tensor_contract(output_tensor, expected_frames=expected_output_frames(segment_idx + 1))
        # save_video_ffmpeg(output_tensor, os.path.join(output_dir, f"video_continue_{segment_idx+1}"), merge_speech_path, fps=save_fps, quality=5)
        # del output_tensor

    # if cp_rank == 0 and os.path.exists(merge_speech_path):
    #     os.remove(merge_speech_path)
    with debug_profile.phase("final_output_contract"):
        validate_output_tensor_contract(output_tensor, expected_frames=expected_output_frames(num_segments))
    # CRITICAL: generated segments include padded silent coverage; return only source-audio frames for lip-sync.
    with debug_profile.phase("trim_to_target_frames"):
        output_tensor = trim_output_tensor_to_target_frames(output_tensor, condition.get("target_output_frames"))
    return output_tensor


# def _parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--input_json',
#         type=str,
#         default='assets/avatar/multi_example_1.json',
#     )
#     parser.add_argument(
#         '--output_dir',
#         type=str,
#         default='./outputs_avatar_multi'
#     )
#     parser.add_argument(
#         '--resolution',
#         type=str,
#         default='480p',
#         choices=['480p', '720p']
#     )
#     parser.add_argument(
#         '--num_segments',
#         type=int,
#         default=1
#     )
#     parser.add_argument(
#         '--num_inference_steps',
#         type=int,
#         default=50
#     )
#     parser.add_argument(
#         '--ref_img_index',
#         type=int,
#         default=10
#     )
#     parser.add_argument(
#         '--mask_frame_range',
#         type=int,
#         default=3
#     )
#     parser.add_argument(
#         '--text_guidance_scale',
#         type=float,
#         default=4.0
#     )
#     parser.add_argument(
#         '--audio_guidance_scale',
#         type=float,
#         default=4.0
#     )

#     parser.add_argument(
#         "--context_parallel_size",
#         type=int,
#         default=1,
#     )
#     parser.add_argument(
#         "--checkpoint_dir",
#         type=str,
#         default="./weights/LongCat-Video-Avatar",
#     )
#     parser.add_argument(
#         "--model_type",
#         type=str,
#         default="avatar-v1.0",
#     )
#     parser.add_argument(
#         "--use_distill",
#         action='store_true',
#     )
#     parser.add_argument(
#         "--use_int8",
#         action='store_true',
#         help="Load INT8 quantized DiT model for reduced VRAM usage"
#     )

#     args = parser.parse_args()

#     return args


# if __name__ == "__main__":
#     args = _parse_args()
#     generate(args)
