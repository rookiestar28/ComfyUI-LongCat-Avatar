import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from LongCat_Video.mlx_generation_backend import MlxRuntimeModules, run_mlx_generation
from LongCat_Video.mlx_runner_contract import MLX_RUNNER_SCHEMA_VERSION, MlxRunnerRequest
from LongCat_Video.mlx_runner_validation import MLX_VARIANT_DIRNAMES


class FakeLinear:
    pass


class FakeArray:
    def __init__(self, frame_count=1):
        self.frame_count = frame_count

    def __truediv__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __add__(self, other):
        return self

    def transpose(self, *args):
        return self

    def __getitem__(self, key):
        return self

    def clip(self, low, high):
        return self

    def astype(self, dtype):
        return self

    def __iter__(self):
        return iter([object()] * self.frame_count)


class FakeModel:
    @classmethod
    def from_config(cls, config):
        instance = cls()
        instance.config = config
        instance.loaded = []
        return instance

    def load_weights(self, path, strict=False):
        self.loaded.append((path, strict))

    def parameters(self):
        return []


class FakeTextEncoder(FakeModel):
    def __call__(self, ids, mask=None):
        return FakeArray()


class FakePipeline:
    def __init__(self, vae, text_encoder, audio_encoder, dit, config):
        self.vae = vae
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.dit = dit
        self.config = config

    def __call__(self, **kwargs):
        return FakeArray(frame_count=kwargs["num_frames"])


class FakeImageObject:
    def __init__(self, width=1, height=1):
        self.width = width
        self.height = height

    def convert(self, mode):
        return self

    def resize(self, size, resample):
        self.width, self.height = size
        return self

    def __array__(self, dtype=None):
        return FakeArray()


class FakeImageModule:
    BICUBIC = 3

    @staticmethod
    def open(path):
        return FakeImageObject()


class FakeWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.frames = 0

    def append_data(self, frame):
        self.frames += 1

    def close(self):
        self.path.write_bytes(b"mp4")


class FakeImageio:
    @staticmethod
    def get_writer(path, **kwargs):
        return FakeWriter(path)


class FakeLibrosa:
    @staticmethod
    def load(path, sr):
        return FakeArray(), sr


class FakeWhisperFeatureExtractor:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, audio, sampling_rate, return_tensors):
        return SimpleNamespace(input_features=FakeArray())


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def __call__(self, prompt, return_tensors, padding, max_length, truncation):
        return SimpleNamespace(
            input_ids=FakeArray(),
            attention_mask=FakeArray(),
        )


class FakeMx:
    @staticmethod
    def array(value):
        return value

    @staticmethod
    def eval(*args):
        return None


class FakeNn:
    Linear = FakeLinear

    @staticmethod
    def quantize(*args, **kwargs):
        return None


class FakeNp:
    float32 = "float32"
    int32 = "int32"
    uint8 = "uint8"

    @staticmethod
    def asarray(value, dtype=None):
        return value if isinstance(value, FakeArray) else FakeArray()

    @staticmethod
    def save(path, value):
        Path(path).write_bytes(b"npy")


def fake_runtime():
    return MlxRuntimeModules(
        mx=FakeMx,
        nn=FakeNn,
        np=FakeNp,
        image=FakeImageModule,
        librosa=FakeLibrosa,
        whisper_feature_extractor=FakeWhisperFeatureExtractor,
        t5_tokenizer_fast=FakeTokenizer,
        imageio=FakeImageio,
        autoencoder_cls=FakeModel,
        dit_cls=FakeModel,
        umt5_cls=FakeTextEncoder,
        whisper_cls=FakeModel,
        pipeline_cls=FakePipeline,
        pipeline_config_cls=lambda: SimpleNamespace(),
    )


class MlxGenerationBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.weights_root = self.root / "weights"
        self.output_dir = self.root / "output"
        self.weights_root.mkdir()
        self.output_dir.mkdir()
        self.image_path = self.root / "portrait.png"
        self.audio_path = self.root / "speech.wav"
        self.image_path.write_bytes(b"image")
        self.audio_path.write_bytes(b"audio")
        self.variant_dir = self.build_variant_layout("q4-merged")

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def touch(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def build_variant_layout(self, variant):
        variant_dir = self.weights_root / MLX_VARIANT_DIRNAMES[variant]
        self.write_json(variant_dir / "vae" / "config.json", {"component": "vae"})
        self.touch(variant_dir / "vae" / "diffusion_pytorch_model.safetensors")
        self.write_json(variant_dir / "text_encoder" / "config.json", {"component": "text_encoder"})
        self.write_json(
            variant_dir / "text_encoder" / "model.safetensors.index.json",
            {"weight_map": {"encoder.block.0": "model-00001-of-00001.safetensors"}},
        )
        self.touch(variant_dir / "text_encoder" / "model-00001-of-00001.safetensors")
        self.write_json(variant_dir / "audio_encoder" / "config.json", {"component": "audio_encoder"})
        self.touch(variant_dir / "audio_encoder" / "model.safetensors")
        self.write_json(
            variant_dir / "dit" / "config.json",
            {"component": "dit", "quantization": {"bits": 4, "group_size": 64}},
        )
        self.write_json(
            variant_dir / "dit" / "diffusion_pytorch_model.safetensors.index.json",
            {"weight_map": {"blocks.0": "diffusion_pytorch_model-00001-of-00001.safetensors"}},
        )
        self.touch(variant_dir / "dit" / "diffusion_pytorch_model-00001-of-00001.safetensors")
        self.write_json(variant_dir / "scheduler" / "scheduler_config.json", {"shift": 7.0})
        self.write_json(variant_dir / "tokenizer" / "tokenizer.json", {"model": "umt5"})
        self.write_json(variant_dir / "tokenizer" / "tokenizer_config.json", {"model_max_length": 512})
        self.write_json(variant_dir / "tokenizer" / "special_tokens_map.json", {"eos_token": "</s>"})
        return variant_dir

    def request(self):
        return MlxRunnerRequest.from_mapping(
            {
                "schema_version": MLX_RUNNER_SCHEMA_VERSION,
                "variant": "q4-merged",
                "weights_root": str(self.weights_root),
                "image_path": str(self.image_path),
                "audio_path": str(self.audio_path),
                "prompt": "prompt",
                "negative_prompt": "",
                "height": 256,
                "width": 432,
                "num_frames": 29,
                "fps": 30,
                "seed": 1,
                "output_dir": str(self.output_dir),
                "output_basename": "smoke",
            }
        )

    def test_run_mlx_generation_writes_frames_and_video_with_fake_runtime(self):
        result = run_mlx_generation(self.request(), variant_dir=self.variant_dir, runtime=fake_runtime())

        self.assertTrue(Path(result["frames_path"]).is_file())
        self.assertTrue(Path(result["video_path"]).is_file())
        self.assertEqual(result["runtime"]["backend"], "mlx")
        self.assertEqual(result["runtime"]["variant"], "q4-merged")
        self.assertIn("load_seconds", result["timings"])
        self.assertIn("inference_seconds", result["timings"])


if __name__ == "__main__":
    unittest.main()
