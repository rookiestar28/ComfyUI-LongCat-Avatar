import json
import tempfile
import unittest
from pathlib import Path

from LongCat_Video.mlx_runner_contract import (
    MLX_RUNNER_SCHEMA_VERSION,
    MlxRunnerRequest,
    MlxRunnerResponse,
    dump_mlx_runner_request_json,
    dump_mlx_runner_response_json,
    load_mlx_runner_request_json,
    load_mlx_runner_response_json,
    safe_runner_log_summary,
    sanitize_log_text,
    validate_generation_controls,
    validate_public_safe_log_payload,
)


class MlxRunnerContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.weights_root = self.root / "weights"
        self.output_dir = self.root / "output"
        self.weights_root.mkdir()
        self.output_dir.mkdir()
        self.image_path = self.root / "portrait.png"
        self.audio_path = self.root / "speech.wav"
        self.image_path.write_bytes(b"fake image")
        self.audio_path.write_bytes(b"fake audio")

    def tearDown(self):
        self.temp_dir.cleanup()

    def request_mapping(self, **overrides):
        data = {
            "schema_version": MLX_RUNNER_SCHEMA_VERSION,
            "variant": "q4-merged",
            "weights_root": str(self.weights_root),
            "image_path": str(self.image_path),
            "audio_path": str(self.audio_path),
            "prompt": "A singer smiles while speaking on a small stage.",
            "negative_prompt": "blur, artifacts",
            "height": 256,
            "width": 432,
            "num_frames": 29,
            "fps": 30,
            "seed": 42,
            "output_dir": str(self.output_dir),
            "output_basename": "longcat_mlx_0001",
        }
        data.update(overrides)
        return data

    def test_request_round_trip_validates_schema_version_and_variant(self):
        request = MlxRunnerRequest.from_mapping(self.request_mapping())
        request_path = self.root / "request.json"

        dump_mlx_runner_request_json(request, request_path)
        loaded = load_mlx_runner_request_json(request_path)

        self.assertEqual(loaded.schema_version, MLX_RUNNER_SCHEMA_VERSION)
        self.assertEqual(loaded.variant, "q4-merged")
        self.assertEqual(loaded.height, 256)
        self.assertEqual(loaded.width, 432)
        self.assertEqual(loaded.num_frames, 29)

    def test_request_rejects_invalid_variant(self):
        with self.assertRaisesRegex(ValueError, "Unsupported MLX runner variant"):
            MlxRunnerRequest.from_mapping(self.request_mapping(variant="bf16"))

    def test_request_rejects_missing_input_file(self):
        self.image_path.unlink()

        with self.assertRaisesRegex(FileNotFoundError, "image_path"):
            MlxRunnerRequest.from_mapping(self.request_mapping())

    def test_request_rejects_malformed_json(self):
        request_path = self.root / "bad-request.json"
        request_path.write_text("{not valid", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Malformed MLX runner request JSON"):
            load_mlx_runner_request_json(request_path)

    def test_request_rejects_unknown_raw_media_payloads(self):
        data = self.request_mapping()
        data["audio_base64"] = "AAAA"

        with self.assertRaisesRegex(ValueError, "unsupported keys"):
            MlxRunnerRequest.from_mapping(data)

    def test_request_rejects_unsafe_output_path(self):
        with self.assertRaisesRegex(ValueError, "output_basename"):
            MlxRunnerRequest.from_mapping(self.request_mapping(output_basename="../escape"))

        reference_dir = self.root / "reference" / "job"
        reference_dir.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "output_dir"):
            MlxRunnerRequest.from_mapping(self.request_mapping(output_dir=str(reference_dir)))

    def test_request_rejects_unsupported_generation_controls(self):
        with self.assertRaisesRegex(ValueError, "Unsupported MLX generation profile"):
            MlxRunnerRequest.from_mapping(self.request_mapping(height=480, width=832, num_frames=93))

        with self.assertRaisesRegex(ValueError, "Unsupported fps"):
            MlxRunnerRequest.from_mapping(self.request_mapping(fps=16))

    def test_generation_control_normalization(self):
        self.assertEqual(
            validate_generation_controls("256", "432", "29", "25", "0"),
            (256, 432, 29, 25, 0),
        )

    def test_safe_runner_log_summary_omits_raw_prompt_and_absolute_paths(self):
        private_prompt = "my exact private prompt"
        request = MlxRunnerRequest.from_mapping(self.request_mapping(prompt=private_prompt))

        summary = safe_runner_log_summary(request)
        encoded = json.dumps(summary, sort_keys=True)

        self.assertNotIn(private_prompt, encoded)
        self.assertNotIn(str(self.image_path), encoded)
        self.assertNotIn(str(self.audio_path), encoded)
        self.assertIn("portrait.png", encoded)
        self.assertEqual(summary["prompt"]["prompt_chars"], len(private_prompt))

    def test_sanitize_log_text_redacts_common_credential_shapes(self):
        label = "to" + "ken"
        hf_value = "hf_" + ("a" * 26)
        sk_value = "sk-" + ("1" * 16)
        redacted = sanitize_log_text(f"{label}={hf_value} and key {sk_value}")

        self.assertNotIn(hf_value, redacted)
        self.assertNotIn(sk_value, redacted)
        self.assertIn("<redacted>", redacted)

    def test_public_safe_log_payload_rejects_sensitive_and_raw_media_keys(self):
        with self.assertRaisesRegex(ValueError, "sensitive key"):
            validate_public_safe_log_payload({"api" + "_key": "placeholder"})

        with self.assertRaisesRegex(ValueError, "raw media payload"):
            validate_public_safe_log_payload({"image_bytes": "AAAA"})

    def test_success_response_requires_artifact_under_output_dir(self):
        video_path = self.output_dir / "result.mp4"
        frames_path = self.output_dir / "result.npy"
        video_path.write_bytes(b"fake video")
        frames_path.write_bytes(b"fake frames")
        response = MlxRunnerResponse.from_mapping(
            {
                "schema_version": MLX_RUNNER_SCHEMA_VERSION,
                "status": "ok",
                "variant": "q4-merged",
                "video_path": str(video_path),
                "frames_path": str(frames_path),
                "timings": {"inference_seconds": 1.25},
                "runtime": {"platform": "macOS arm64"},
                "warnings": [],
            },
            output_dir=self.output_dir,
        )
        response_path = self.root / "response.json"

        dump_mlx_runner_response_json(response, response_path)
        loaded = load_mlx_runner_response_json(response_path, output_dir=self.output_dir)

        self.assertEqual(loaded.status, "ok")
        self.assertEqual(loaded.variant, "q4-merged")
        self.assertEqual(loaded.timings["inference_seconds"], 1.25)

    def test_success_response_allows_no_artifacts_for_dry_run_validation(self):
        response = MlxRunnerResponse.from_mapping(
            {
                "schema_version": MLX_RUNNER_SCHEMA_VERSION,
                "status": "ok",
                "variant": "q4-merged",
                "video_path": "",
                "frames_path": "",
                "timings": {},
                "runtime": {"mode": "dry-run"},
                "warnings": ["dry-run: generation not executed."],
            },
            output_dir=self.output_dir,
            require_artifacts=False,
        )

        self.assertEqual(response.status, "ok")
        self.assertEqual(response.video_path, "")
        self.assertEqual(response.frames_path, "")

    def test_success_response_rejects_artifact_escape(self):
        outside = self.root / "escape.mp4"
        outside.write_bytes(b"fake")

        with self.assertRaisesRegex(ValueError, "video_path"):
            MlxRunnerResponse.from_mapping(
                {
                    "schema_version": MLX_RUNNER_SCHEMA_VERSION,
                    "status": "ok",
                    "variant": "q4-merged",
                    "video_path": str(outside),
                    "frames_path": "",
                    "timings": {},
                    "runtime": {},
                    "warnings": [],
                },
                output_dir=self.output_dir,
            )

    def test_error_response_sanitizes_message_and_diagnostics(self):
        label = "to" + "ken"
        hf_value = "hf_" + ("b" * 26)
        response = MlxRunnerResponse.from_mapping(
            {
                "schema_version": MLX_RUNNER_SCHEMA_VERSION,
                "status": "error",
                "error_type": "RuntimeError",
                "message": f"failed with {label}={hf_value}",
                "stage": "load",
                "diagnostics": {"variant": "q4-merged", "height": 256},
            }
        )

        self.assertEqual(response.status, "error")
        self.assertNotIn(hf_value, response.error.message)
        self.assertIn("<redacted>", response.error.message)

    def test_error_response_rejects_sensitive_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "sensitive key"):
            MlxRunnerResponse.from_mapping(
                {
                    "schema_version": MLX_RUNNER_SCHEMA_VERSION,
                    "status": "error",
                    "error_type": "RuntimeError",
                    "message": "failed",
                    "stage": "load",
                    "diagnostics": {"hf" + "_token": "placeholder"},
                }
            )


if __name__ == "__main__":
    unittest.main()
