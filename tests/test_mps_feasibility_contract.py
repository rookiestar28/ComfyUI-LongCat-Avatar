import unittest

from LongCat_Video.performance_contract import (
    MPS_EXPERIMENTAL_BRANCH,
    build_mps_feasibility_report,
    require_cuda_device,
)


class MPSFeasibilityContractTests(unittest.TestCase):
    def test_mps_probe_reports_dedicated_branch_and_blockers(self):
        report = build_mps_feasibility_report(requested_device="mps", recommended_dtype="fp16", attention_backend="sdpa")

        self.assertEqual(report.target_branch, MPS_EXPERIMENTAL_BRANCH)
        self.assertFalse(report.supported)
        self.assertEqual(report.requested_device, "mps")
        self.assertEqual(report.recommended_dtype, "fp16")
        self.assertEqual(report.attention_backend, "sdpa")
        self.assertGreaterEqual(len(report.cuda_only_assumptions), 4)
        self.assertGreaterEqual(len(report.initialization_blockers), 3)
        self.assertIn("Do not merge MPS inference into main", report.non_merge_condition)

    def test_mps_still_fails_fast_on_current_runtime_contract(self):
        with self.assertRaisesRegex(RuntimeError, "CPU and MPS are not supported"):
            require_cuda_device("mps")


if __name__ == "__main__":
    unittest.main()
