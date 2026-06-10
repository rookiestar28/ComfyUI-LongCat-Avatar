from __future__ import annotations

import math
from typing import Any


def parse_audio_crop_time(value: str | int | float, *, field_name: str) -> float:
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{field_name} must not be empty.")
        if ":" not in text:
            try:
                seconds = float(text)
            except ValueError as exc:
                raise ValueError(f"{field_name} must be seconds, MM:SS, or HH:MM:SS.") from exc
        else:
            parts = text.split(":")
            if len(parts) not in (2, 3) or any(part.strip() == "" for part in parts):
                raise ValueError(f"{field_name} must be seconds, MM:SS, or HH:MM:SS.")
            try:
                numeric_parts = [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"{field_name} must be seconds, MM:SS, or HH:MM:SS.") from exc
            if len(parts) == 2:
                minutes, seconds_part = numeric_parts
                seconds = minutes * 60 + seconds_part
            else:
                hours, minutes, seconds_part = numeric_parts
                seconds = hours * 3600 + minutes * 60 + seconds_part

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{field_name} must be a non-negative finite time.")
    return seconds


def crop_audio_payload(audio: dict[str, Any], start_time: str = "0:00", end_time: str = "1:00") -> dict[str, Any]:
    if not isinstance(audio, dict):
        raise TypeError("AudioCrop: audio must be a ComfyUI AUDIO payload.")
    if "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("AudioCrop: audio payload must contain waveform and sample_rate.")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError("AudioCrop: sample_rate must be greater than 0.")

    shape = getattr(waveform, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError("AudioCrop: waveform must expose a sample dimension.")
    sample_count = int(shape[-1])
    if sample_count < 0:
        raise ValueError("AudioCrop: waveform sample dimension is invalid.")

    start_seconds = parse_audio_crop_time(start_time, field_name="start_time")
    end_seconds = parse_audio_crop_time(end_time, field_name="end_time")
    start_frame = min(max(int(math.floor(start_seconds * sample_rate)), 0), sample_count)
    end_frame = min(max(int(math.floor(end_seconds * sample_rate)), 0), sample_count)

    if start_frame > end_frame:
        raise ValueError("AudioCrop: start_time must be less than or equal to end_time after clamping.")

    return {
        "waveform": waveform[..., start_frame:end_frame],
        "sample_rate": sample_rate,
    }
