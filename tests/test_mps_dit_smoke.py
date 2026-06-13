import types
import unittest
from pathlib import Path

from LongCat_Video.mps_dit_smoke import STATUS_BLOCKED, STATUS_PASS, run_mps_dit_attention_smoke


class FakeMPSBackend:
    def __init__(self, *, available=True):
        self.available = available

    def is_available(self):
        return self.available

    def is_built(self):
        return True


class FakeMPSOps:
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


class FakeDit:
    def __init__(self, *, fail_forward=False):
        self.config = types.SimpleNamespace(
            caption_channels=4096,
            audio_window=5,
            audio_block=12,
            audio_channel=768,
            vae_scale=4,
        )
        self.fail_forward = fail_forward
        self.to_calls = []
        self.calls = []

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_forward:
            raise RuntimeError("scaled_dot_product_attention mps blocker")
        latent = kwargs["hidden_states"]
        return FakeTensor(shape=latent.shape, device=latent.device, dtype="float32")


class MPSDitSmokeTests(unittest.TestCase):
    def test_success_records_dit_boundary_metadata(self):
        fake_torch = FakeTorch()
        fake_dit = FakeDit()

        result = run_mps_dit_attention_smoke(
            dit_model=fake_dit,
            model_source="injected_int8",
            torch_module=fake_torch,
            environ={"PYTORCH_ENABLE_MPS_FALLBACK": "0"},
        )

        self.assertEqual(result.status, STATUS_PASS)
        self.assertEqual(result.stage, "complete")
        self.assertTrue(result.native_mps)
        self.assertEqual(result.model_source, "injected_int8")
        self.assertEqual(result.attention_backend, "sdpa")
        self.assertEqual(result.dtype_policy["dit"], "bf16")
        self.assertEqual(result.boundary_tensors["latent"]["shape"], [1, 16, 2, 2, 2])
        self.assertEqual(result.boundary_tensors["prompt"]["dtype"], "bfloat16")
        self.assertEqual(result.boundary_tensors["negative_prompt"]["shape"], [1, 1, 1, 4096])
        self.assertEqual(result.boundary_tensors["audio"]["shape"], [1, 5, 5, 12, 768])
        self.assertEqual(result.output_shape, (1, 16, 2, 2, 2))
        self.assertEqual(fake_dit.to_calls, [((), {"device": "mps"})])
        self.assertEqual(len(fake_dit.calls), 1)
        self.assertIsNone(fake_dit.calls[0]["encoder_attention_mask"])
        self.assertTrue(any(snapshot.label == "after_forward" for snapshot in result.memory))

    def test_fallback_enabled_blocks_before_model_execution(self):
        fake_dit = FakeDit()

        result = run_mps_dit_attention_smoke(
            dit_model=fake_dit,
            torch_module=FakeTorch(),
            environ={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
        )

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.stage, "environment")
        self.assertTrue(result.cpu_fallback_enabled)
        self.assertEqual(fake_dit.calls, [])

    def test_mps_rejects_non_sdpa_attention_before_model_execution(self):
        fake_dit = FakeDit()

        result = run_mps_dit_attention_smoke(
            dit_model=fake_dit,
            attention_mode="auto",
            torch_module=FakeTorch(),
            environ={},
        )

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.stage, "contract")
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertIn("not supported on MPS", result.detail)
        self.assertEqual(fake_dit.calls, [])

    def test_forward_failure_records_stage_boundary_and_traceback(self):
        result = run_mps_dit_attention_smoke(
            dit_model=FakeDit(fail_forward=True),
            torch_module=FakeTorch(),
            environ={},
        )

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.stage, "forward")
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertIn("scaled_dot_product_attention", result.detail)
        self.assertIn("latent", result.boundary_tensors)
        self.assertTrue(any("test_mps_dit_smoke.py" in frame for frame in result.traceback_location))

    def test_mps_unavailable_blocks_before_model_execution(self):
        fake_dit = FakeDit()

        result = run_mps_dit_attention_smoke(
            dit_model=fake_dit,
            torch_module=FakeTorch(mps_available=False),
            environ={},
        )

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.stage, "backend")
        self.assertIn("MPS backend is not available", result.detail)
        self.assertEqual(fake_dit.calls, [])

    def test_avatar_dit_uses_backend_aware_fp32_modulation_context(self):
        avatar_source = Path("LongCat_Video/longcat_video/modules/avatar/longcat_video_dit_avatar.py").read_text(
            encoding="utf-8"
        )
        blocks_source = Path("LongCat_Video/longcat_video/modules/blocks.py").read_text(encoding="utf-8")

        self.assertIn("fp32_modulation_context", avatar_source)
        self.assertNotIn("amp.autocast", avatar_source)
        self.assertNotIn("device_type='cuda'", avatar_source)
        self.assertIn("def fp32_modulation_context", blocks_source)

    def test_quantized_dit_loader_applies_attention_mode_to_config(self):
        source = Path("LongCat_Video/longcat_video/modules/quantization.py").read_text(encoding="utf-8")

        self.assertIn("apply_attention_mode_to_config", source)
        self.assertIn("config = apply_attention_mode_to_config(config, attention_mode)", source)
        self.assertLess(
            source.index("config = apply_attention_mode_to_config(config, attention_mode)"),
            source.index("print_attention_diagnostics(attention_mode, config)"),
        )


if __name__ == "__main__":
    unittest.main()
