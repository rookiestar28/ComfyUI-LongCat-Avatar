# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import logging
import comfy.model_management as mm
from PIL import Image
import numpy as np
from comfy.utils import common_upscale
import folder_paths
import soundfile as sf
import uuid
from LongCat_Video.backend_capabilities import empty_cache, read_memory_stats
cur_path = os.path.dirname(os.path.abspath(__file__))

def audio2path(audio,):
    unique_id = uuid.uuid4().hex[:8]
    audio_file = os.path.join(folder_paths.get_temp_directory(), f"audio_refer_temp_{unique_id}.wav")
    waveform = audio["waveform"].squeeze(0)
    waveform_np = waveform.cpu().numpy() if hasattr(waveform, 'cpu') else waveform.numpy()

    # 3. 格式转换：torchaudio 格式为 (channels, samples)，soundfile 需要 (samples, channels)
    # 如果是单声道音频 (1, samples)，转置后变成 (samples, 1)，符合 soundfile 的单声道要求
    if waveform_np.ndim == 2:
        waveform_np = waveform_np.T

    sf.write(audio_file, waveform_np, audio["sample_rate"])

    return audio_file
def auto_match(num_frames):
    vae_scale_factor_temporal = 4
    k = round((num_frames + vae_scale_factor_temporal - 1) / (vae_scale_factor_temporal * vae_scale_factor_temporal))
    k = max(k, 1)
    corrected_num_frames = (vae_scale_factor_temporal * vae_scale_factor_temporal) * k - (vae_scale_factor_temporal - 1)
    if corrected_num_frames != num_frames:
        print(f"[LongCat Video] Auto-corrected num_frames from {num_frames} to {corrected_num_frames} to satisfy temporal alignment (16k - 3).")
    num_frames = corrected_num_frames
    return num_frames

def get_runtime_device():
    return mm.get_torch_device()


def _runtime_empty_cache():
    empty_cache(get_runtime_device(), torch_module=torch)


def _runtime_max_memory_allocated():
    stats = read_memory_stats(get_runtime_device(), torch_module=torch)
    return stats.max_allocated_bytes or stats.allocated_bytes or 0


def clear_comfyui_cache(unload_loaded_models=False):
    if unload_loaded_models:
        for pipe in mm.loaded_models():
            try:
                pipe.unpatch_model(device_to=torch.device("cpu"))
            except Exception as exc:
                logging.warning("LongCat cache cleanup could not unpatch a loaded model: %s", exc)
    mm.soft_empty_cache()
    _runtime_empty_cache()
    max_gpu_memory = _runtime_max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")

def gc_cleanup():
    gc.collect()
    _runtime_empty_cache()


def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor2image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def tensor2pillist_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensor2list(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list

def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "bilinear", "center")
    samples = samples.movedim(1, -1)
    return samples

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "bilinear", "center")
    samples = img.movedim(1, -1)
    img = tensor2image(samples)
    return img


def map_0_1_to_neg1_1(t):

    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.float()

    try:
        vmax = float(t.max())
    except Exception:
        vmax = 1.0
    if vmax > 2.0:
        t = t / 255.0
    try:
        vmin = float(t.min())
        vmax = float(t.max())
    except Exception:
        vmin, vmax = -1.0, 1.0
    if vmin >= 0.0 and vmax <= 1.1:
        t = t * 2.0 - 1.0
    return t

def map_neg1_1_to_0_1(t):
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.float()
    t = (t + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    return t
