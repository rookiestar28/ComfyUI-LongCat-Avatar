import json
import sys
import tempfile
import unittest
from pathlib import Path

from LongCat_Video.mlx_runner_validation import (
    MLX_RUNNER_DEPENDENCIES,
    MLX_VARIANT_DIRNAMES,
    build_mlx_environment_report,
    validate_mlx_preflight,
    validate_mlx_weights_root,
)


def _dependency_probe(*, missing=()):
    missing = set(missing)
    return {
        requirement.import_name: {
            "available": requirement.import_name not in missing,
            "version": "1.0.0" if requirement.import_name not in missing else "",
            "error": "ImportError" if requirement.import_name in missing else "",
        }
        for requirement in MLX_RUNNER_DEPENDENCIES
    }


class MlxRunnerValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.weights_root = self.root / "weights"
        self.weights_root.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def touch(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def build_variant_layout(self, variant, *, quant_bits=None, group_size=64):
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

        dit_config = {"component": "dit"}
        if quant_bits is not None:
            dit_config["quantization"] = {
                "bits": quant_bits,
                "group_size": group_size,
                "skip_patterns": ["final_layer.linear"],
            }
        self.write_json(variant_dir / "dit" / "config.json", dit_config)
        self.write_json(
            variant_dir / "dit" / "diffusion_pytorch_model.safetensors.index.json",
            {"weight_map": {"blocks.0.attn": "diffusion_pytorch_model-00001-of-00001.safetensors"}},
        )
        self.touch(variant_dir / "dit" / "diffusion_pytorch_model-00001-of-00001.safetensors")

        self.write_json(variant_dir / "scheduler" / "scheduler_config.json", {"shift": 7.0})
        self.write_json(variant_dir / "tokenizer" / "tokenizer.json", {"model": "umt5"})
        self.write_json(variant_dir / "tokenizer" / "tokenizer_config.json", {"model_max_length": 512})
        self.write_json(variant_dir / "tokenizer" / "special_tokens_map.json", {"eos_token": "</s>"})
        return variant_dir

    def test_environment_report_classifies_16gb_apple_silicon_as_probe_only(self):
        report = build_mlx_environment_report(
            {
                "executable_exists": True,
                "python_executable": sys.executable,
                "python_version": "3.12.0",
                "platform_system": "Darwin",
                "platform_machine": "arm64",
                "unified_memory_bytes": 16 * 1024**3,
                "dependencies": _dependency_probe(),
            },
            runner_python=sys.executable,
            variant="q4-merged",
        )

        self.assertTrue(report.is_macos)
        self.assertTrue(report.is_arm64)
        self.assertTrue(report.mlx_available)
        self.assertEqual(report.support_status, "install_probe_only")
        self.assertTrue(any("16 GB Apple Silicon" in item for item in report.warnings))

    def test_environment_report_records_missing_dependency_action(self):
        report = build_mlx_environment_report(
            {
                "executable_exists": True,
                "python_executable": sys.executable,
                "python_version": "3.12.0",
                "platform_system": "Darwin",
                "platform_machine": "arm64",
                "unified_memory_bytes": 32 * 1024**3,
                "dependencies": _dependency_probe(missing={"transformers"}),
            },
            runner_python=sys.executable,
            variant="q4-merged",
        )

        self.assertFalse(report.is_generation_candidate)
        self.assertTrue(any("transformers" in issue for issue in report.issues))
        self.assertTrue(any("Install missing MLX runner dependency" in issue for issue in report.issues))

    def test_environment_report_requires_external_longcat_mlx_package(self):
        report = build_mlx_environment_report(
            {
                "executable_exists": True,
                "python_executable": sys.executable,
                "python_version": "3.12.0",
                "platform_system": "Darwin",
                "platform_machine": "arm64",
                "unified_memory_bytes": 32 * 1024**3,
                "dependencies": _dependency_probe(missing={"longcat_video_avatar"}),
            },
            runner_python=sys.executable,
            variant="q4-merged",
        )

        self.assertFalse(report.is_generation_candidate)
        self.assertTrue(any("longcat-video-avatar-mlx" in issue for issue in report.issues))

    def test_weights_root_accepts_complete_q4_layout(self):
        self.build_variant_layout("q4-merged", quant_bits=4)

        report = validate_mlx_weights_root(self.weights_root, "q4-merged")

        self.assertTrue(report.is_complete)
        self.assertEqual(report.root_mode, "weights_root_parent")
        self.assertEqual(report.quantization["bits"], 4)
        self.assertFalse(report.issues)

    def test_weights_root_accepts_direct_variant_dir_with_warning(self):
        variant_dir = self.build_variant_layout("q4-merged", quant_bits=4)

        report = validate_mlx_weights_root(variant_dir, "q4-merged")

        self.assertTrue(report.is_complete)
        self.assertEqual(report.root_mode, "variant_dir")
        self.assertTrue(any("parent weights root is preferred" in warning for warning in report.warnings))

    def test_q8_weight_validation_rejects_wrong_quantization_bits(self):
        self.build_variant_layout("q8-merged", quant_bits=4)

        report = validate_mlx_weights_root(self.weights_root, "q8-merged")

        self.assertFalse(report.is_complete)
        self.assertTrue(any("expected 8-bit DiT quantization" in issue for issue in report.issues))

    def test_merged_weight_validation_rejects_quantization_metadata(self):
        self.build_variant_layout("merged", quant_bits=4)

        report = validate_mlx_weights_root(self.weights_root, "merged")

        self.assertFalse(report.is_complete)
        self.assertTrue(any("should not include DiT quantization" in issue for issue in report.issues))

    def test_missing_shard_report_uses_relative_public_safe_path(self):
        variant_dir = self.build_variant_layout("q4-merged", quant_bits=4)
        (variant_dir / "text_encoder" / "model-00001-of-00001.safetensors").unlink()

        report = validate_mlx_weights_root(self.weights_root, "q4-merged")
        encoded = "\n".join(report.issues)

        self.assertFalse(report.is_complete)
        self.assertIn("text_encoder/model-00001-of-00001.safetensors", encoded)
        self.assertNotIn(str(self.weights_root), encoded)

    def test_preflight_combines_mocked_environment_and_weight_reports(self):
        self.build_variant_layout("q4-merged", quant_bits=4)

        def fake_probe(runner_python, script, timeout_seconds):
            self.assertIn("requirements", script)
            self.assertEqual(runner_python, sys.executable)
            self.assertGreater(timeout_seconds, 0)
            return json.dumps(
                {
                    "executable_exists": True,
                    "python_executable": runner_python,
                    "python_version": "3.12.0",
                    "platform_system": "Darwin",
                    "platform_machine": "arm64",
                    "dependencies": _dependency_probe(),
                }
            )

        report = validate_mlx_preflight(
            runner_python=sys.executable,
            weights_root=self.weights_root,
            variant="q4-merged",
            run_probe=fake_probe,
            unified_memory_bytes=32 * 1024**3,
        )

        self.assertTrue(report.is_ready_for_generation)
        self.assertFalse(report.issues)


if __name__ == "__main__":
    unittest.main()
