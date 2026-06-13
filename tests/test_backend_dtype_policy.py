import unittest
from pathlib import Path
from types import SimpleNamespace

from LongCat_Video.backend_dtype_policy import (
    PRECISION_BF16,
    PRECISION_FP16,
    PRECISION_FP32,
    mps_safe_numeric_dtype_name,
    randn_for_device,
    resolve_random_generator_device,
    resolve_backend_dtype_policy,
)


class FakeTensor:
    def __init__(self, *, device):
        self.device = device
        self.to_calls = []

    def to(self, **kwargs):
        self.to_calls.append(kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        return self

    def __add__(self, other):
        return FakeTensor(device=self.device)

    def __matmul__(self, other):
        return FakeTensor(device=self.device)


class FakeConv3d:
    def __init__(self, *args, **kwargs):
        self.to_calls = []

    def to(self, **kwargs):
        self.to_calls.append(kwargs)
        return self

    def __call__(self, value):
        return FakeTensor(device=value.device)


class FakeMPSBackend:
    def __init__(self, *, available=True):
        self.available = available

    def is_available(self):
        return self.available

    def is_built(self):
        return True


class FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    def __init__(self, *, fail_bfloat16=False):
        self.backends = SimpleNamespace(mps=FakeMPSBackend())
        self.fail_bfloat16 = fail_bfloat16
        self.randn_calls = []
        self.empty_calls = []
        self.ones_calls = []
        self.arange_calls = []
        self.nn = SimpleNamespace(Conv3d=FakeConv3d)

    def empty(self, shape, *, device=None, dtype=None):
        if str(device).startswith("mps") and dtype == self.bfloat16 and self.fail_bfloat16:
            raise TypeError("BFloat16 is not supported on MPS")
        self.empty_calls.append((shape, str(device), dtype))
        return FakeTensor(device=device)

    def ones(self, shape, *, device=None, dtype=None):
        if str(device).startswith("mps") and dtype == self.bfloat16 and self.fail_bfloat16:
            raise TypeError("BFloat16 is not supported on MPS")
        self.ones_calls.append((shape, str(device), dtype))
        return FakeTensor(device=device or "cpu")

    def arange(self, end, *, device=None, dtype=None):
        if str(device).startswith("mps") and dtype == self.bfloat16 and self.fail_bfloat16:
            raise RuntimeError("arange_mps not implemented for BFloat16")
        self.arange_calls.append((end, str(device), dtype))
        return FakeTensor(device=device)

    def randn(self, shape, *, generator=None, device=None, dtype=None):
        self.randn_calls.append((shape, generator, device, dtype))
        return FakeTensor(device=device)


class BackendDTypePolicyTests(unittest.TestCase):
    def test_cuda_auto_precision_preserves_bfloat16_contract(self):
        fake_torch = FakeTorch()

        policy = resolve_backend_dtype_policy("cuda:0", torch_module=fake_torch)

        self.assertEqual(policy.backend, "cuda")
        self.assertEqual(policy.text_encoder_precision, PRECISION_BF16)
        self.assertEqual(policy.audio_encoder_precision, PRECISION_BF16)
        self.assertEqual(policy.dit_precision, PRECISION_BF16)
        self.assertEqual(policy.vae_precision, PRECISION_BF16)
        self.assertEqual(policy.math_precision, PRECISION_FP32)
        self.assertEqual(policy.text_encoder_dtype, "bfloat16")

    def test_mps_auto_precision_prefers_bfloat16_models_when_probe_passes(self):
        fake_torch = FakeTorch()

        policy = resolve_backend_dtype_policy("mps", torch_module=fake_torch)

        self.assertEqual(policy.backend, "mps")
        self.assertEqual(policy.text_encoder_precision, PRECISION_BF16)
        self.assertEqual(policy.audio_encoder_precision, PRECISION_FP32)
        self.assertEqual(policy.dit_precision, PRECISION_BF16)
        self.assertEqual(policy.vae_precision, PRECISION_BF16)
        self.assertEqual(policy.math_precision, PRECISION_FP32)
        self.assertEqual(policy.text_encoder_dtype, "bfloat16")
        self.assertIsNotNone(policy.bfloat16_probe)
        self.assertTrue(policy.bfloat16_probe.supported)
        self.assertIn("conv3d", policy.bfloat16_probe.detail)

    def test_mps_auto_precision_falls_back_to_fp16_when_bfloat16_probe_fails(self):
        fake_torch = FakeTorch(fail_bfloat16=True)

        policy = resolve_backend_dtype_policy("mps", torch_module=fake_torch)

        self.assertEqual(policy.text_encoder_precision, PRECISION_FP16)
        self.assertEqual(policy.audio_encoder_precision, PRECISION_FP32)
        self.assertEqual(policy.dit_precision, PRECISION_FP16)
        self.assertEqual(policy.vae_precision, PRECISION_FP16)
        self.assertEqual(policy.math_precision, PRECISION_FP32)
        self.assertIsNotNone(policy.bfloat16_probe)
        self.assertFalse(policy.bfloat16_probe.supported)
        self.assertIn("bf16 probe failed", policy.reason)

    def test_mps_bfloat16_request_requires_runtime_probe(self):
        fake_torch = FakeTorch(fail_bfloat16=True)

        with self.assertRaisesRegex(ValueError, "runtime probe failed"):
            resolve_backend_dtype_policy("mps", requested_precision="bf16", torch_module=fake_torch)

    def test_mps_bfloat16_request_can_pass_when_probe_passes(self):
        fake_torch = FakeTorch()

        policy = resolve_backend_dtype_policy("mps", requested_precision="bf16", torch_module=fake_torch)

        self.assertEqual(policy.text_encoder_precision, PRECISION_BF16)
        self.assertIsNotNone(policy.bfloat16_probe)
        self.assertTrue(policy.bfloat16_probe.supported)

    def test_unsupported_precision_requests_fail_fast(self):
        fake_torch = FakeTorch()

        with self.assertRaisesRegex(ValueError, "Unsupported precision"):
            resolve_backend_dtype_policy("mps", requested_precision="fp8", torch_module=fake_torch)
        with self.assertRaisesRegex(ValueError, "bf16-only"):
            resolve_backend_dtype_policy("cuda:0", requested_precision="fp16", torch_module=fake_torch)

    def test_mps_safe_numeric_dtype_downcasts_float64_and_int64(self):
        self.assertEqual(mps_safe_numeric_dtype_name("mps", "torch.float64"), "float32")
        self.assertEqual(mps_safe_numeric_dtype_name("mps", "double"), "float32")
        self.assertEqual(mps_safe_numeric_dtype_name("mps", "torch.int64"), "int32")
        self.assertEqual(mps_safe_numeric_dtype_name("cuda:0", "torch.float64"), "float64")

    def test_random_generator_device_uses_cpu_for_mps(self):
        self.assertEqual(resolve_random_generator_device("mps"), "cpu")
        self.assertEqual(resolve_random_generator_device("mps:0"), "cpu")
        self.assertEqual(resolve_random_generator_device("cuda:0"), "cuda:0")
        self.assertEqual(resolve_random_generator_device("cpu"), "cpu")

    def test_randn_for_mps_generates_on_cpu_then_moves_to_mps(self):
        fake_torch = FakeTorch()

        tensor = randn_for_device((1, 2), generator="seeded", device="mps", dtype="float16", torch_module=fake_torch)

        self.assertEqual(fake_torch.randn_calls, [((1, 2), "seeded", "cpu", "float16")])
        self.assertEqual(tensor.to_calls, [{"device": "mps"}])
        self.assertEqual(tensor.device, "mps")

    def test_randn_for_cuda_preserves_target_device_generation(self):
        fake_torch = FakeTorch()

        tensor = randn_for_device((1, 2), generator="seeded", device="cuda:0", dtype="bfloat16", torch_module=fake_torch)

        self.assertEqual(fake_torch.randn_calls, [((1, 2), "seeded", "cuda:0", "bfloat16")])
        self.assertEqual(tensor.to_calls, [])
        self.assertEqual(tensor.device, "cuda:0")

    def test_scheduler_and_rope_sources_keep_mps_float32_numeric_contract(self):
        scheduler_source = Path("LongCat_Video/longcat_video/modules/scheduling_flow_match_euler_discrete.py").read_text(
            encoding="utf-8"
        )
        avatar_rope_source = Path("LongCat_Video/longcat_video/modules/avatar/rope_3d.py").read_text(encoding="utf-8")
        base_rope_source = Path("LongCat_Video/longcat_video/modules/rope_3d.py").read_text(encoding="utf-8")

        self.assertIn('sample.device.type == "mps"', scheduler_source)
        self.assertIn("dtype=torch.float32", scheduler_source)
        self.assertIn("np.float32", avatar_rope_source)
        self.assertIn("freqs_cis.float().to", avatar_rope_source)
        self.assertIn("np.float32", base_rope_source)
        self.assertIn("freqs_cis.float().to", base_rope_source)

    def test_avatar_pipeline_uses_backend_rng_helper(self):
        source = Path("LongCat_Video/longcat_video/pipeline_longcat_video_avatar.py").read_text(encoding="utf-8")

        self.assertIn("randn_for_device", source)
        self.assertNotIn("latents = torch.randn(shape", source)

    def test_avatar_generation_wrappers_use_backend_generator_device_helper(self):
        for path in (
            Path("LongCat_Video/run_demo_avatar_single_audio_to_video.py"),
            Path("LongCat_Video/run_demo_avatar_multi_audio_to_video.py"),
        ):
            with self.subTest(path=str(path)):
                source = path.read_text(encoding="utf-8")

                self.assertIn("resolve_random_generator_device", source)
                self.assertIn("torch.Generator(device=resolve_random_generator_device(device))", source)
                self.assertNotIn("torch.Generator(device=device)", source)


if __name__ == "__main__":
    unittest.main()
