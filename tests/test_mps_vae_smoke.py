import tempfile
import types
import unittest
from pathlib import Path

from LongCat_Video.mps_vae_smoke import STATUS_BLOCKED, STATUS_PASS, run_mps_vae_encode_decode_smoke


class FakeMPSBackend:
    def __init__(self, *, available=True):
        self.available = available

    def is_available(self):
        return self.available

    def is_built(self):
        return True


class FakeMPSOps:
    def empty_cache(self):
        return None

    def synchronize(self):
        return None

    def current_allocated_memory(self):
        return 1_000_000

    def driver_allocated_memory(self):
        return 2_000_000

    def recommended_max_memory(self):
        return 3_000_000


class FakeTensor:
    def __init__(self, *, shape, device, dtype):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.to_calls = []

    def to(self, **kwargs):
        self.to_calls.append(kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]
        return self

    def __add__(self, other):
        return FakeTensor(shape=self.shape, device=self.device, dtype=self.dtype)

    def __matmul__(self, other):
        return FakeTensor(shape=self.shape, device=self.device, dtype=self.dtype)


class FakeConv3d:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def to(self, **kwargs):
        return self

    def __call__(self, value):
        return FakeTensor(shape=value.shape, device=value.device, dtype=value.dtype)


class FakeContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    __version__ = "fake"
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    def __init__(self, *, mps_available=True):
        self.backends = types.SimpleNamespace(mps=FakeMPSBackend(available=mps_available))
        self.mps = FakeMPSOps()
        self.nn = types.SimpleNamespace(Conv3d=FakeConv3d)
        self.zeros_calls = []

    def empty(self, shape, *, device=None, dtype=None):
        return FakeTensor(shape=shape, device=device, dtype=dtype)

    def ones(self, shape, *, device=None, dtype=None):
        return FakeTensor(shape=shape, device=device or "cpu", dtype=dtype)

    def arange(self, end, *, device=None, dtype=None):
        return FakeTensor(shape=(end,), device=device, dtype=dtype)

    def zeros(self, shape, *, device=None, dtype=None):
        self.zeros_calls.append((shape, device, dtype))
        return FakeTensor(shape=shape, device=device, dtype=dtype)

    def inference_mode(self):
        return FakeContext()


class FakeLatentDistribution:
    def __init__(self, tensor):
        self.tensor = tensor

    def mode(self):
        return self.tensor


class FakeVAE:
    fail_encode = False
    instances = []

    def __init__(self, *, torch_dtype):
        self.torch_dtype = torch_dtype
        self.to_calls = []
        self.loaded_state = None
        FakeVAE.instances.append(self)

    @classmethod
    def load_config(cls, path):
        return {"path": path}

    @classmethod
    def from_config(cls, config, torch_dtype=None):
        return cls(torch_dtype=torch_dtype)

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_state = (state_dict, strict)
        return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self

    def encode(self, sample):
        if self.fail_encode:
            raise RuntimeError("slow_conv2d_forward_mps unsupported dtype")
        latent = FakeTensor(shape=(1, 16, 1, 8, 8), device=sample.device, dtype=sample.dtype)
        return types.SimpleNamespace(latent_dist=FakeLatentDistribution(latent))

    def decode(self, latent, return_dict=True):
        decoded = FakeTensor(shape=(1, 3, 1, 64, 64), device=latent.device, dtype=latent.dtype)
        return types.SimpleNamespace(sample=decoded)


class MPSVaeSmokeTests(unittest.TestCase):
    def setUp(self):
        FakeVAE.fail_encode = False
        FakeVAE.instances = []

    def test_success_records_dtype_shapes_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, weights = self._write_fake_files(tmp)
            safe_load_calls = []

            def fake_safe_load(path, device=None):
                safe_load_calls.append((Path(path).name, device))
                return {"weight": object()}

            fake_torch = FakeTorch()

            result = run_mps_vae_encode_decode_smoke(
                vae_config_path=config,
                vae_weights_path=weights,
                torch_module=fake_torch,
                environ={"PYTORCH_ENABLE_MPS_FALLBACK": "0"},
                safe_load_fn=fake_safe_load,
                autoencoder_cls=FakeVAE,
            )

        self.assertEqual(result.status, STATUS_PASS)
        self.assertEqual(result.stage, "complete")
        self.assertTrue(result.native_mps)
        self.assertEqual(result.dtype_policy["vae"], "bf16")
        self.assertEqual(result.latent_shape, (1, 16, 1, 8, 8))
        self.assertEqual(result.decoded_shape, (1, 3, 1, 64, 64))
        self.assertEqual(safe_load_calls, [("diffusion_pytorch_model.safetensors", "cpu")])
        self.assertEqual(fake_torch.zeros_calls, [((1, 3, 1, 64, 64), "mps", "bfloat16")])
        self.assertTrue(any(snapshot.label == "before_load" for snapshot in result.memory))
        self.assertTrue(any(snapshot.label == "after_decode" for snapshot in result.memory))
        self.assertEqual(FakeVAE.instances[0].to_calls, [((), {"device": "mps", "dtype": "bfloat16"})])

    def test_fallback_enabled_blocks_native_mps_evidence_before_loading(self):
        safe_load_calls = []

        result = run_mps_vae_encode_decode_smoke(
            vae_config_path="missing-config.json",
            vae_weights_path="missing-weights.safetensors",
            torch_module=FakeTorch(),
            environ={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
            safe_load_fn=lambda *args, **kwargs: safe_load_calls.append((args, kwargs)),
            autoencoder_cls=FakeVAE,
        )

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.stage, "environment")
        self.assertTrue(result.cpu_fallback_enabled)
        self.assertFalse(result.native_mps)
        self.assertEqual(safe_load_calls, [])

    def test_missing_weights_records_file_blocker_without_full_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text("{}", encoding="utf-8")
            missing_weights = Path(tmp) / "missing.safetensors"

            result = run_mps_vae_encode_decode_smoke(
                vae_config_path=config,
                vae_weights_path=missing_weights,
                torch_module=FakeTorch(),
                environ={},
                safe_load_fn=lambda *args, **kwargs: {},
                autoencoder_cls=FakeVAE,
            )

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.error_type, "FileNotFoundError")
        self.assertIn("missing.safetensors", result.detail)
        self.assertNotIn(str(missing_weights.parent), result.detail)

    def test_encode_failure_records_stage_and_traceback_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, weights = self._write_fake_files(tmp)
            FakeVAE.fail_encode = True

            result = run_mps_vae_encode_decode_smoke(
                vae_config_path=config,
                vae_weights_path=weights,
                torch_module=FakeTorch(),
                environ={},
                safe_load_fn=lambda *args, **kwargs: {"weight": object()},
                autoencoder_cls=FakeVAE,
            )

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.stage, "encode")
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertIn("slow_conv2d_forward_mps", result.detail)
        self.assertTrue(result.traceback_location)
        self.assertTrue(any("test_mps_vae_smoke.py" in frame for frame in result.traceback_location))

    def test_mps_unavailable_blocks_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, weights = self._write_fake_files(tmp)

            result = run_mps_vae_encode_decode_smoke(
                vae_config_path=config,
                vae_weights_path=weights,
                torch_module=FakeTorch(mps_available=False),
                environ={},
                safe_load_fn=lambda *args, **kwargs: {"weight": object()},
                autoencoder_cls=FakeVAE,
            )

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.stage, "backend")
        self.assertIn("MPS backend is not available", result.detail)
        self.assertEqual(FakeVAE.instances, [])

    def _write_fake_files(self, directory):
        config = Path(directory) / "config.json"
        weights = Path(directory) / "diffusion_pytorch_model.safetensors"
        config.write_text("{}", encoding="utf-8")
        weights.write_bytes(b"fake")
        return config, weights


if __name__ == "__main__":
    unittest.main()
