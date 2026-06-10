from .wav2vec2 import Wav2Vec2ModelWrapper
from transformers import Wav2Vec2FeatureExtractor, WhisperModel, AutoFeatureExtractor,WhisperConfig
import os
import torch
from safetensors.torch import load_file as safe_load_file
from ...audio_contract import (
    normalize_longcat_avatar_whisper_state_dict,
    validate_longcat_avatar_whisper_load_result,
)
def get_audio_encoder(checkpoint_path, model_type="avatar-v1.0",config_dir=""):
    if model_type == "avatar-v1.0":
        model = Wav2Vec2ModelWrapper(checkpoint_path)
        model.feature_extractor._freeze_parameters()
        return model
    if model_type == "avatar-v1.5":
        #model = WhisperModel.from_pretrained(checkpoint_path).eval()
        config_=WhisperConfig.from_pretrained(config_dir)
        model=WhisperModel(config_)
        sd=(
            torch.load(checkpoint_path, map_location="cpu")
            if checkpoint_path.endswith(".pt")
            else safe_load_file(checkpoint_path, device="cpu")
        )
        sd=normalize_longcat_avatar_whisper_state_dict(sd)
        # CRITICAL: strict=False must not hide Whisper key-prefix mismatches; random audio encoder weights break lip sync.
        load_result=model.load_state_dict(sd, strict=False)
        validate_longcat_avatar_whisper_load_result(load_result)
        del sd
        model.eval()
        model.requires_grad_(False)
        return model

def get_audio_feature_extractor(checkpoint_path, model_type="avatar-v1.0",):
    if model_type == "avatar-v1.0":
        return Wav2Vec2FeatureExtractor(checkpoint_path, local_files_only=True)
    if model_type == "avatar-v1.5":
        return AutoFeatureExtractor.from_pretrained(checkpoint_path)
