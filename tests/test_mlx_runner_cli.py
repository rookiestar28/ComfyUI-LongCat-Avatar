import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from LongCat_Video.mlx_runner_cli import (
    MlxRunnerArtifacts,
    MlxRunnerOptions,
    main,
    run_mlx_runner,
)
from LongCat_Video.mlx_runner_contract import (
    MLX_RUNNER_SCHEMA_VERSION,
    load_mlx_runner_response_json,
)
from LongCat_Video.mlx_runner_validation import (
    MLX_RUNNER_DEPENDENCIES,
    MLX_VARIANT_DIRNAMES,
    build_mlx_environment_report,
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


class MlxRunnerCliTests(unittest.TestCase):
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
        self.build_variant_layout("q4-merged", quant_bits=4)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def touch(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def build_variant_layout(self, variant, *, quant_bits=None):
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
                "group_size": 64,
                "skip_patterns": ["final_layer.linear"],
            }
        self.write_json(variant_dir / "dit" / "config.json", dit_config)
        self.write_json(
            variant_dir / "dit" / "diffusion_pytorch_model.safetensors.index.json",
            {"weight_map": {"blocks.0": "diffusion_pytorch_model-00001-of-00001.safetensors"}},
        )
        self.touch(variant_dir / "dit" / "diffusion_pytorch_model-00001-of-00001.safetensors")
        self.write_json(variant_dir / "scheduler" / "scheduler_config.json", {"shift": 7.0})
        self.write_json(variant_dir / "tokenizer" / "tokenizer.json", {"model": "umt5"})
        self.write_json(variant_dir / "tokenizer" / "tokenizer_config.json", {"model_max_length": 512})
        self.write_json(variant_dir / "tokenizer" / "special_tokens_map.json", {"eos_token": "</s>"})

    def request_mapping(self, **overrides):
        data = {
            "schema_version": MLX_RUNNER_SCHEMA_VERSION,
            "variant": "q4-merged",
            "weights_root": str(self.weights_root),
            "image_path": str(self.image_path),
            "audio_path": str(self.audio_path),
            "prompt": "private scene prompt that must not appear in default logs",
            "negative_prompt": "blur, artifacts",
            "height": 256,
            "width": 432,
            "num_frames": 29,
            "fps": 30,
            "seed": 123,
            "output_dir": str(self.output_dir),
            "output_basename": "longcat_mlx_job",
        }
        data.update(overrides)
        return data

    def write_request(self, **overrides):
        request_path = self.root / "request.json"
        self.write_json(request_path, self.request_mapping(**overrides))
        return request_path

    def response_path(self):
        return self.output_dir / "response.json"

    def environment_report(self, *, missing=()):
        return build_mlx_environment_report(
            {
                "executable_exists": True,
                "python_executable": sys.executable,
                "python_version": "3.12.0",
                "platform_system": "Darwin",
                "platform_machine": "arm64",
                "unified_memory_bytes": 32 * 1024**3,
                "dependencies": _dependency_probe(missing=missing),
            },
            runner_python=sys.executable,
            variant="q4-merged",
        )

    def test_main_dry_run_writes_success_without_artifacts(self):
        request_path = self.write_request()
        response_path = self.response_path()

        code = main(["--request", str(request_path), "--response", str(response_path), "--mode", "dry-run"])

        self.assertEqual(code, 0)
        response = load_mlx_runner_response_json(response_path, output_dir=self.output_dir, require_artifacts=False)
        self.assertEqual(response.status, "ok")
        self.assertEqual(response.video_path, "")
        self.assertTrue(any("dry-run" in warning for warning in response.warnings))

    def test_generate_backend_success_writes_artifacts_under_output_dir(self):
        request_path = self.write_request()
        response_path = self.response_path()

        def backend(request, weights, environment):
            video_path = Path(request.output_dir) / "result.mp4"
            frames_path = Path(request.output_dir) / "result.npy"
            video_path.write_bytes(b"video")
            frames_path.write_bytes(b"frames")
            return MlxRunnerArtifacts(
                video_path=str(video_path),
                frames_path=str(frames_path),
                timings={"inference_seconds": 1.0},
                runtime={"backend": "mock"},
            )

        code = run_mlx_runner(
            MlxRunnerOptions(str(request_path), str(response_path), mode="generate"),
            generation_backend=backend,
            environment_report=self.environment_report(),
        )

        self.assertEqual(code, 0)
        response = load_mlx_runner_response_json(response_path, output_dir=self.output_dir)
        self.assertEqual(response.status, "ok")
        self.assertEqual(response.runtime["backend"], "mock")

    def test_frame_artifact_success_when_mp4_export_fails(self):
        request_path = self.write_request()
        response_path = self.response_path()

        def backend(request, weights, environment):
            frames_path = Path(request.output_dir) / "result.npy"
            frames_path.write_bytes(b"frames")
            return MlxRunnerArtifacts(
                frames_path=str(frames_path),
                warnings=("mp4 export failed; frame artifact retained.",),
            )

        code = run_mlx_runner(
            MlxRunnerOptions(str(request_path), str(response_path), mode="generate"),
            generation_backend=backend,
            environment_report=self.environment_report(),
        )

        self.assertEqual(code, 0)
        response = load_mlx_runner_response_json(response_path, output_dir=self.output_dir)
        self.assertEqual(response.video_path, "")
        self.assertTrue(response.frames_path.endswith("result.npy"))
        self.assertTrue(any("frame artifact retained" in warning for warning in response.warnings))

    def test_backend_artifact_escape_writes_error_response(self):
        request_path = self.write_request()
        response_path = self.response_path()
        outside = self.root / "escape.mp4"
        outside.write_bytes(b"video")

        def backend(request, weights, environment):
            return MlxRunnerArtifacts(video_path=str(outside))

        code = run_mlx_runner(
            MlxRunnerOptions(str(request_path), str(response_path), mode="generate"),
            generation_backend=backend,
            environment_report=self.environment_report(),
        )

        self.assertEqual(code, 1)
        response = load_mlx_runner_response_json(response_path, output_dir=self.output_dir, require_artifacts=False)
        self.assertEqual(response.status, "error")
        self.assertIn("video_path", response.error.message)

    def test_dependency_failure_blocks_backend(self):
        request_path = self.write_request()
        response_path = self.response_path()
        called = False

        def backend(request, weights, environment):
            nonlocal called
            called = True
            return MlxRunnerArtifacts()

        code = run_mlx_runner(
            MlxRunnerOptions(str(request_path), str(response_path), mode="generate"),
            generation_backend=backend,
            environment_report=self.environment_report(missing={"mlx"}),
        )

        self.assertEqual(code, 1)
        self.assertFalse(called)
        response = load_mlx_runner_response_json(response_path)
        self.assertEqual(response.error.stage, "probe")
        self.assertIn("environment validation failed", response.error.message)

    def test_malformed_request_writes_error_response(self):
        request_path = self.root / "bad-request.json"
        request_path.write_text("{not valid", encoding="utf-8")
        response_path = self.response_path()

        code = run_mlx_runner(MlxRunnerOptions(str(request_path), str(response_path), mode="dry-run"))

        self.assertEqual(code, 1)
        response = load_mlx_runner_response_json(response_path)
        self.assertEqual(response.status, "error")
        self.assertEqual(response.error.stage, "schema")

    def test_timeout_error_writes_bounded_error_response(self):
        request_path = self.write_request()
        response_path = self.response_path()

        def backend(request, weights, environment):
            raise TimeoutError("timed out while waiting for MLX worker")

        code = run_mlx_runner(
            MlxRunnerOptions(str(request_path), str(response_path), mode="generate"),
            generation_backend=backend,
            environment_report=self.environment_report(),
        )

        self.assertEqual(code, 1)
        response = load_mlx_runner_response_json(response_path)
        self.assertEqual(response.error.error_type, "TimeoutError")
        self.assertEqual(response.error.stage, "inference")

    def test_safe_summary_does_not_print_raw_prompt(self):
        prompt = "private prompt text for output safety"
        request_path = self.write_request(prompt=prompt)
        response_path = self.response_path()
        output = io.StringIO()

        code = run_mlx_runner(
            MlxRunnerOptions(str(request_path), str(response_path), mode="dry-run", log_summary=True),
            output_stream=output,
        )

        self.assertEqual(code, 0)
        self.assertNotIn(prompt, output.getvalue())
        self.assertIn("prompt_chars", output.getvalue())


if __name__ == "__main__":
    unittest.main()
