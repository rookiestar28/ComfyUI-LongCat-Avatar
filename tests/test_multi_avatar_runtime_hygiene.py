from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@contextmanager
def loaded_multi_runtime():
    module_names = [
        "PIL",
        "PIL.Image",
        "numpy",
        "torch",
        "transformers",
        "diffusers",
        "diffusers.utils",
        "librosa",
        "soundfile",
        "LongCat_Video.longcat_video.pipeline_longcat_video_avatar",
        "LongCat_Video.longcat_video.modules.scheduling_flow_match_euler_discrete",
        "LongCat_Video.longcat_video.modules.autoencoder_kl_wan",
        "LongCat_Video.longcat_video.modules.avatar.longcat_video_dit_avatar",
        "LongCat_Video.longcat_video.modules.quantization",
        "LongCat_Video.longcat_video.audio_process",
        "LongCat_Video.longcat_video.audio_process.torch_utils",
    ]
    old_modules = {name: sys.modules.get(name) for name in module_names}
    previous_target = sys.modules.pop("LongCat_Video.run_demo_avatar_multi_audio_to_video", None)
    try:
        sys.modules["PIL"] = _module("PIL")
        sys.modules["PIL.Image"] = _module("PIL.Image")
        sys.modules["numpy"] = _module(
            "numpy",
            float32="float32",
            uint8="uint8",
            array=lambda value, dtype=None: value,
            zeros_like=lambda value: [0.0 for _ in value],
            concatenate=lambda values: [item for value in values for item in value],
            append=lambda value, values: list(value) + list(values),
        )
        sys.modules["torch"] = _module(
            "torch",
            cuda=types.SimpleNamespace(empty_cache=lambda: None, ipc_collect=lambda: None),
        )
        sys.modules["transformers"] = _module(
            "transformers",
            AutoTokenizer=object,
            UMT5EncoderModel=object,
        )
        sys.modules["diffusers"] = _module("diffusers")
        sys.modules["diffusers.utils"] = _module("diffusers.utils", load_image=lambda path: None)
        sys.modules["librosa"] = _module("librosa", load=lambda *args, **kwargs: ([0.0], 16000))
        sys.modules["soundfile"] = _module("soundfile", write=lambda *args, **kwargs: None)
        sys.modules["LongCat_Video.longcat_video.pipeline_longcat_video_avatar"] = _module(
            "LongCat_Video.longcat_video.pipeline_longcat_video_avatar",
            LongCatVideoAvatarPipeline=object,
        )
        sys.modules["LongCat_Video.longcat_video.modules.scheduling_flow_match_euler_discrete"] = _module(
            "LongCat_Video.longcat_video.modules.scheduling_flow_match_euler_discrete",
            FlowMatchEulerDiscreteScheduler=object,
        )
        sys.modules["LongCat_Video.longcat_video.modules.autoencoder_kl_wan"] = _module(
            "LongCat_Video.longcat_video.modules.autoencoder_kl_wan",
            AutoencoderKLWan=object,
        )
        sys.modules["LongCat_Video.longcat_video.modules.avatar.longcat_video_dit_avatar"] = _module(
            "LongCat_Video.longcat_video.modules.avatar.longcat_video_dit_avatar",
            LongCatVideoAvatarTransformer3DModel=object,
        )
        sys.modules["LongCat_Video.longcat_video.modules.quantization"] = _module(
            "LongCat_Video.longcat_video.modules.quantization",
            load_quantized_dit=lambda *args, **kwargs: None,
        )
        sys.modules["LongCat_Video.longcat_video.audio_process"] = _module(
            "LongCat_Video.longcat_video.audio_process",
            get_audio_encoder=lambda *args, **kwargs: None,
            get_audio_feature_extractor=lambda *args, **kwargs: None,
        )
        sys.modules["LongCat_Video.longcat_video.audio_process.torch_utils"] = _module(
            "LongCat_Video.longcat_video.audio_process.torch_utils",
            save_video_ffmpeg=lambda *args, **kwargs: None,
        )
        yield importlib.import_module("LongCat_Video.run_demo_avatar_multi_audio_to_video")
    finally:
        sys.modules.pop("LongCat_Video.run_demo_avatar_multi_audio_to_video", None)
        if previous_target is not None:
            sys.modules["LongCat_Video.run_demo_avatar_multi_audio_to_video"] = previous_target
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class MultiAvatarRuntimeHygieneTests(unittest.TestCase):
    def test_extract_vocal_from_speech_uses_structured_file_move(self):
        with loaded_multi_runtime() as module, tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.wav"
            target = tmp / "target.wav"
            vocal_dir = tmp / "vocals"
            vocal_dir.mkdir()
            moved_source = vocal_dir / "vocals.wav"
            moved_source.write_text("voice", encoding="utf-8")

            class Separator:
                def separate(self, path):
                    self.path = path
                    return ["vocals.wav"]

            result = module.extract_vocal_from_speech(source, target, Separator(), tmp)

            self.assertEqual(Path(result), target)
            self.assertEqual(target.read_text(encoding="utf-8"), "voice")
            self.assertFalse(moved_source.exists())

    def test_audio_prepare_multi_rejects_parallel_length_mismatch_before_merge(self):
        with loaded_multi_runtime() as module:
            values = {
                "left_temp.wav": [1.0, 2.0],
                "right_temp.wav": [3.0, 4.0, 5.0],
                "left_raw.wav": [1.0, 2.0],
                "right_raw.wav": [3.0, 4.0, 5.0],
            }

            def fake_load(path, sr):
                return values[path], sr

            module.librosa.load = fake_load

            with self.assertRaisesRegex(ValueError, "equal-length"):
                module.audio_prepare_multi(
                    "left_temp.wav",
                    "right_temp.wav",
                    generate_duration=1.0,
                    left_raw_speech_path="left_raw.wav",
                    right_raw_speech_path="right_raw.wav",
                    audio_type="para",
                )

    def test_partial_person_bbox_rejected_with_value_error(self):
        with loaded_multi_runtime() as module:
            with self.assertRaisesRegex(ValueError, "both left and right person boxes"):
                module.resolve_person_bbox_coordinates(
                    1280,
                    720,
                    [0, 0, 100, 100],
                    None,
                )

    def test_default_person_bbox_split_matches_existing_dual_person_layout(self):
        with loaded_multi_runtime() as module:
            self.assertEqual(
                module.resolve_person_bbox_coordinates(1280, 720, None, None),
                (72, 64, 648, 576, 72, 704, 648, 1216),
            )


if __name__ == "__main__":
    unittest.main()
