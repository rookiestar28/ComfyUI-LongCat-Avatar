import types
import unittest

from LongCat_Video.mps_smoke_matrix import STATUS_BLOCKED, STATUS_SKIPPED, run_mps_smoke_matrix


class FakeMPSBackend:
    def __init__(self, available):
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
        return 1_000

    def driver_allocated_memory(self):
        return 2_000

    def recommended_max_memory(self):
        return 3_000


class FakeTorch:
    __version__ = "fake"
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    def __init__(self, *, mps_available):
        self.backends = types.SimpleNamespace(mps=FakeMPSBackend(mps_available))
        self.mps = FakeMPSOps()

    def empty(self, shape, *, device=None, dtype=None):
        return object()


class MPSSmokeMatrixTests(unittest.TestCase):
    def test_mps_unavailable_blocks_matrix_and_components(self):
        result = run_mps_smoke_matrix(
            torch_module=FakeTorch(mps_available=False),
            environ={},
            branch="macos-mps",
            commit="abc1234",
        )
        public = result.to_public_dict()

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertIn("MPS backend is not available.", result.blockers)
        self.assertEqual(public["environment"]["branch"], "macos-mps")
        self.assertEqual(public["environment"]["commit"], "abc1234")
        self.assertTrue(any(step["name"] == "component_model_load" for step in public["steps"]))

    def test_fallback_enabled_cannot_satisfy_native_mps_evidence(self):
        result = run_mps_smoke_matrix(
            torch_module=FakeTorch(mps_available=True),
            environ={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
            branch="macos-mps",
            commit="abc1234",
        )
        steps = {step.name: step for step in result.steps}

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertIn("PYTORCH_ENABLE_MPS_FALLBACK is enabled", result.blockers[0])
        self.assertEqual(steps["attention_probe"].status, STATUS_SKIPPED)
        self.assertFalse(steps["attention_probe"].native_mps)
        self.assertTrue(steps["attention_probe"].cpu_fallback_enabled)

    def test_available_mps_still_blocks_without_real_model_components(self):
        result = run_mps_smoke_matrix(
            torch_module=FakeTorch(mps_available=True),
            environ={},
            branch="macos-mps",
            commit="abc1234",
        )
        steps = {step.name: step for step in result.steps}

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(steps["mps_available"].status, "pass")
        self.assertEqual(steps["dtype_policy"].status, "pass")
        self.assertEqual(steps["cache_sync_memory"].status, "pass")
        self.assertEqual(steps["component_minimal_generation"].status, STATUS_BLOCKED)


if __name__ == "__main__":
    unittest.main()
