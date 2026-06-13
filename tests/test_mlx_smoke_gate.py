import unittest

from LongCat_Video.mlx_smoke_gate import evaluate_mlx_smoke_gate


class MlxSmokeGateTests(unittest.TestCase):
    def evidence(self, **overrides):
        data = {
            "schema_version": 1,
            "status": "passed",
            "variant": "q4-merged",
            "height": 256,
            "width": 432,
            "num_frames": 29,
            "host_system": "Darwin",
            "host_machine": "arm64",
            "unified_memory_gb": 32,
            "response_json_valid": True,
            "response_status": "ok",
            "artifact_present": True,
            "artifact_kind": "mp4",
            "timings": {"inference_seconds": 101.5},
        }
        data.update(overrides)
        return data

    def test_accepts_first_q4_artifact_gate(self):
        decision = evaluate_mlx_smoke_gate(self.evidence())

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.support_status, "accepted_mlx_external_runner")
        self.assertIn("MLX external runner", decision.public_support_label)
        self.assertNotIn("PyTorch MPS", decision.public_support_label)
        self.assertEqual(decision.legacy_mps_status, "blocked")

    def test_rejects_480p_before_first_smoke_gate(self):
        decision = evaluate_mlx_smoke_gate(self.evidence(height=480, width=832, num_frames=93))

        self.assertFalse(decision.accepted)
        self.assertTrue(any("256 x 432 x 29" in issue for issue in decision.issues))

    def test_rejects_16gb_apple_silicon_host(self):
        decision = evaluate_mlx_smoke_gate(self.evidence(unified_memory_gb=16))

        self.assertFalse(decision.accepted)
        self.assertTrue(any("32 GB+" in issue for issue in decision.issues))
        self.assertEqual(decision.legacy_mps_status, "blocked")

    def test_rejects_missing_artifact(self):
        decision = evaluate_mlx_smoke_gate(
            self.evidence(artifact_present=False, artifact_kind="none")
        )

        self.assertFalse(decision.accepted)
        self.assertTrue(any("MP4 or frame artifact" in issue for issue in decision.issues))

    def test_failed_smoke_keeps_support_blocked(self):
        decision = evaluate_mlx_smoke_gate(
            self.evidence(status="failed", response_json_valid=True, artifact_present=False, artifact_kind="none")
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.support_status, "blocked_pending_mlx_q4_artifact")
        self.assertTrue(any("support wording remains blocked" in warning for warning in decision.warnings))

    def test_rejects_non_macos_arm64(self):
        decision = evaluate_mlx_smoke_gate(self.evidence(host_system="Windows", host_machine="AMD64"))

        self.assertFalse(decision.accepted)
        self.assertTrue(any("macOS on Apple Silicon" in issue for issue in decision.issues))

    def test_rejects_invalid_response_json_status(self):
        decision = evaluate_mlx_smoke_gate(self.evidence(response_json_valid=False, response_status="error"))

        self.assertFalse(decision.accepted)
        self.assertTrue(any("valid ok response JSON" in issue for issue in decision.issues))


if __name__ == "__main__":
    unittest.main()
