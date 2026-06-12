import types
import unittest

from LongCat_Video.backend_capabilities import (
    describe_backend,
    device_supports_non_blocking,
    empty_cache,
    format_memory_fields,
    mps_cpu_fallback_enabled,
    normalize_backend_type,
    probe_bfloat16_support,
    read_memory_stats,
    synchronize,
)


class FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.empty_cache_calls = 0
        self.synchronize_calls = []

    def is_available(self):
        return self.available

    def empty_cache(self):
        self.empty_cache_calls += 1

    def synchronize(self, device=None):
        self.synchronize_calls.append(device)

    def memory_allocated(self):
        return 1_000_000_000

    def memory_reserved(self):
        return 2_000_000_000

    def max_memory_allocated(self):
        return 3_000_000_000


class FakeMPSBackend:
    def __init__(self, *, available=True, built=True):
        self.available = available
        self.built = built

    def is_available(self):
        return self.available

    def is_built(self):
        return self.built


class FakeMPSOps:
    def __init__(self):
        self.empty_cache_calls = 0
        self.synchronize_calls = 0

    def empty_cache(self):
        self.empty_cache_calls += 1

    def synchronize(self):
        self.synchronize_calls += 1

    def current_allocated_memory(self):
        return 4_000_000_000

    def driver_allocated_memory(self):
        return 5_000_000_000

    def recommended_max_memory(self):
        return 6_000_000_000


class FakeTorch:
    bfloat16 = "bfloat16"

    def __init__(self, *, mps_available=True, mps_built=True, fail_mps_bfloat16=False, with_mps_ops=True):
        self.cuda = FakeCuda()
        self.backends = types.SimpleNamespace(mps=FakeMPSBackend(available=mps_available, built=mps_built))
        self.mps = FakeMPSOps() if with_mps_ops else types.SimpleNamespace()
        self.fail_mps_bfloat16 = fail_mps_bfloat16
        self.empty_calls = []

    def empty(self, shape, *, device=None, dtype=None):
        if str(device).startswith("mps") and self.fail_mps_bfloat16:
            raise TypeError("BFloat16 is not supported on MPS")
        self.empty_calls.append((shape, str(device), dtype))
        return object()


class BackendCapabilitiesTests(unittest.TestCase):
    def test_normalizes_backend_and_non_blocking_policy(self):
        self.assertEqual(normalize_backend_type("cuda:0"), "cuda")
        self.assertEqual(normalize_backend_type("mps"), "mps")
        self.assertEqual(normalize_backend_type(types.SimpleNamespace(type="cpu", index=None)), "cpu")
        self.assertTrue(device_supports_non_blocking("cuda:0"))
        self.assertFalse(device_supports_non_blocking("mps"))
        self.assertFalse(device_supports_non_blocking("cpu"))

    def test_reports_mps_capabilities_and_fallback_env_without_treating_fallback_as_support(self):
        fake_torch = FakeTorch()

        caps = describe_backend(
            "mps",
            torch_module=fake_torch,
            environ={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
        )

        self.assertEqual(caps.backend, "mps")
        self.assertTrue(caps.available)
        self.assertTrue(caps.built)
        self.assertFalse(caps.supports_non_blocking)
        self.assertTrue(caps.bfloat16.supported)
        self.assertTrue(caps.mps_cpu_fallback_enabled)

    def test_mps_cache_sync_and_memory_use_guarded_mps_apis(self):
        fake_torch = FakeTorch()

        cache_result = empty_cache("mps", torch_module=fake_torch)
        sync_result = synchronize("mps", torch_module=fake_torch)
        stats = read_memory_stats("mps", torch_module=fake_torch)

        self.assertTrue(cache_result.success)
        self.assertTrue(sync_result.success)
        self.assertEqual(fake_torch.mps.empty_cache_calls, 1)
        self.assertEqual(fake_torch.mps.synchronize_calls, 1)
        self.assertEqual(stats.allocated_bytes, 4_000_000_000)
        self.assertEqual(stats.driver_allocated_bytes, 5_000_000_000)
        self.assertEqual(stats.recommended_max_bytes, 6_000_000_000)
        self.assertEqual(
            format_memory_fields(stats),
            ["mps_alloc_gb=4.00", "mps_driver_gb=5.00", "mps_recommended_max_gb=6.00"],
        )

    def test_missing_mps_apis_return_unavailable_diagnostics(self):
        fake_torch = FakeTorch(with_mps_ops=False)

        cache_result = empty_cache("mps", torch_module=fake_torch)
        sync_result = synchronize("mps", torch_module=fake_torch)
        stats = read_memory_stats("mps", torch_module=fake_torch)

        self.assertFalse(cache_result.success)
        self.assertIn("empty_cache API is unavailable", cache_result.detail)
        self.assertFalse(sync_result.success)
        self.assertIn("synchronize API is unavailable", sync_result.detail)
        self.assertFalse(stats.available)
        self.assertIn("current_allocated_memory unavailable", stats.detail)

    def test_mps_bfloat16_probe_does_not_assume_support(self):
        fake_torch = FakeTorch(fail_mps_bfloat16=True)

        result = probe_bfloat16_support("mps", torch_module=fake_torch)

        self.assertFalse(result.supported)
        self.assertIn("BFloat16 is not supported on MPS", result.detail)

    def test_cuda_memory_fields_preserve_existing_debug_names(self):
        fake_torch = FakeTorch()

        cache_result = empty_cache("cuda:0", torch_module=fake_torch)
        sync_result = synchronize("cuda:0", torch_module=fake_torch)
        stats = read_memory_stats("cuda:0", torch_module=fake_torch)

        self.assertTrue(cache_result.success)
        self.assertTrue(sync_result.success)
        self.assertEqual(fake_torch.cuda.empty_cache_calls, 1)
        self.assertEqual(fake_torch.cuda.synchronize_calls, ["cuda:0"])
        self.assertEqual(
            format_memory_fields(stats),
            ["cuda_alloc_gb=1.00", "cuda_reserved_gb=2.00", "cuda_max_alloc_gb=3.00"],
        )

    def test_cpu_has_no_backend_cache_or_memory_api(self):
        fake_torch = FakeTorch()

        self.assertFalse(empty_cache("cpu", torch_module=fake_torch).success)
        self.assertFalse(synchronize("cpu", torch_module=fake_torch).success)
        self.assertFalse(read_memory_stats("cpu", torch_module=fake_torch).available)
        self.assertFalse(mps_cpu_fallback_enabled({"PYTORCH_ENABLE_MPS_FALLBACK": "0"}))


if __name__ == "__main__":
    unittest.main()
