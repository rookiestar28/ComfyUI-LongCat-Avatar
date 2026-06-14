import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from LongCat_Video.mlx_bridge import MlxBridgeSubprocessResult, _run_subprocess, run_mlx_bridge_job
from LongCat_Video.mlx_runner_cli import MlxRunnerOptions, run_mlx_runner
from LongCat_Video.mlx_runner_contract import MLX_RUNNER_SCHEMA_VERSION, dump_mlx_runner_response_json, MlxRunnerResponse
from LongCat_Video.mlx_runner_validation import MLX_VARIANT_DIRNAMES


class MlxBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output_root = self.root / "output"
        self.weights_root = self.root / "weights"
        self.output_root.mkdir()
        self.weights_root.mkdir()
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
            dit_config["quantization"] = {"bits": quant_bits, "group_size": 64}
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

    def image_writer(self, image, path):
        Path(path).write_bytes(b"image")

    def audio_writer(self, audio, path):
        Path(path).write_bytes(b"audio")

    def bridge_kwargs(self, **overrides):
        data = {
            "runner_python": sys.executable,
            "weights_root": self.weights_root,
            "variant": "q4-merged",
            "image": object(),
            "audio": object(),
            "prompt": "private bridge prompt",
            "negative_prompt": "blur",
            "height": 256,
            "width": 432,
            "num_frames": 29,
            "fps": 30,
            "seed": 5,
            "output_root": self.output_root,
            "output_basename": "longcat_mlx",
            "mode": "dry-run",
            "timeout_seconds": 30,
            "retain_job_dir": True,
            "image_writer": self.image_writer,
            "audio_writer": self.audio_writer,
            "job_id": "job001",
        }
        data.update(overrides)
        return data

    def test_bridge_writes_inputs_request_and_launches_with_arg_list(self):
        captured = {}

        def runner(args, timeout_seconds):
            captured["args"] = args
            captured["timeout"] = timeout_seconds
            request_path = Path(args[args.index("--request") + 1])
            response_path = Path(args[args.index("--response") + 1])
            mode = args[args.index("--mode") + 1]
            code = run_mlx_runner(MlxRunnerOptions(str(request_path), str(response_path), mode=mode))
            return MlxBridgeSubprocessResult(code)

        result = run_mlx_bridge_job(**self.bridge_kwargs(subprocess_runner=runner))

        self.assertEqual(result.video_path, "")
        self.assertEqual(captured["args"][0], sys.executable)
        self.assertEqual(captured["args"][1], "-u")
        self.assertEqual(captured["args"][2], "-m")
        self.assertIn("LongCat_Video.mlx_runner_cli", captured["args"])
        self.assertEqual(captured["timeout"], 30.0)
        self.assertTrue(Path(result.request_path).is_file())
        self.assertTrue(Path(result.response_path).is_file())
        self.assertTrue((Path(result.job_dir) / "input.png").is_file())
        self.assertTrue((Path(result.job_dir) / "input.wav").is_file())
        self.assertTrue(Path(result.job_dir).resolve().is_relative_to(self.output_root.resolve()))

    def test_bridge_default_runner_returns_job_log_path(self):
        result = run_mlx_bridge_job(**self.bridge_kwargs(job_id="default_log"))

        self.assertTrue(result.log_path)
        self.assertTrue(Path(result.log_path).is_file())
        self.assertTrue(Path(result.log_path).resolve().is_relative_to(Path(result.job_dir).resolve()))

    def test_default_subprocess_streams_log_before_process_exit(self):
        log_path = self.root / "streaming.log"
        release_path = self.root / "release.txt"
        script = (
            "import pathlib, time\n"
            "print('live-line', flush=True)\n"
            f"release = pathlib.Path({str(release_path)!r})\n"
            "deadline = time.time() + 10\n"
            "while time.time() < deadline and not release.exists():\n"
            "    time.sleep(0.05)\n"
            "print('done-line', flush=True)\n"
        )
        result = {}

        def invoke_runner():
            result["value"] = _run_subprocess(
                [sys.executable, "-u", "-c", script],
                8.0,
                log_path=log_path,
            )

        thread = threading.Thread(target=invoke_runner)
        thread.start()
        saw_live_line = False
        deadline = time.time() + 5
        while time.time() < deadline:
            if log_path.is_file() and "live-line" in log_path.read_text(encoding="utf-8"):
                saw_live_line = True
                break
            time.sleep(0.05)

        self.assertTrue(saw_live_line)
        self.assertTrue(thread.is_alive())
        release_path.write_text("release", encoding="utf-8")
        thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result["value"].returncode, 0)
        self.assertIn("done-line", log_path.read_text(encoding="utf-8"))

    def test_default_subprocess_timeout_preserves_sanitized_log_tail(self):
        log_path = self.root / "timeout.log"
        script = "import time\nprint('before-timeout', flush=True)\ntime.sleep(30)\n"

        with self.assertRaises(subprocess.TimeoutExpired) as context:
            _run_subprocess([sys.executable, "-u", "-c", script], 1.0, log_path=log_path)

        self.assertIn("before-timeout", log_path.read_text(encoding="utf-8"))
        self.assertIn("before-timeout", str(context.exception.output))

    def test_bridge_timeout_cleans_job_when_retention_disabled(self):
        def runner(args, timeout_seconds):
            raise subprocess.TimeoutExpired(args, timeout_seconds)

        with self.assertRaisesRegex(TimeoutError, "timed out"):
            run_mlx_bridge_job(
                **self.bridge_kwargs(subprocess_runner=runner, retain_job_dir=False, job_id="timeout")
            )

        self.assertFalse((self.output_root / "longcat_mlx_timeout").exists())

    def test_bridge_missing_response_json_is_error(self):
        def runner(args, timeout_seconds):
            return MlxBridgeSubprocessResult(0)

        with self.assertRaisesRegex(FileNotFoundError, "response.json"):
            run_mlx_bridge_job(**self.bridge_kwargs(subprocess_runner=runner, job_id="missing"))

    def test_bridge_malformed_response_json_is_error(self):
        def runner(args, timeout_seconds):
            response_path = Path(args[args.index("--response") + 1])
            response_path.write_text("{bad", encoding="utf-8")
            return MlxBridgeSubprocessResult(0)

        with self.assertRaisesRegex(ValueError, "Malformed MLX runner response JSON"):
            run_mlx_bridge_job(**self.bridge_kwargs(subprocess_runner=runner, job_id="malformed"))

    def test_bridge_nonzero_exit_without_response_is_error(self):
        def runner(args, timeout_seconds):
            return MlxBridgeSubprocessResult(2, stderr="failed without response")

        with self.assertRaisesRegex(RuntimeError, "exited without response JSON"):
            run_mlx_bridge_job(**self.bridge_kwargs(subprocess_runner=runner, job_id="nonzero"))

    def test_bridge_rejects_unsafe_returned_artifact_path(self):
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"video")

        def runner(args, timeout_seconds):
            response_path = Path(args[args.index("--response") + 1])
            response = MlxRunnerResponse(
                schema_version=MLX_RUNNER_SCHEMA_VERSION,
                status="ok",
                variant="q4-merged",
                video_path=str(outside),
            )
            dump_mlx_runner_response_json(response, response_path)
            return MlxBridgeSubprocessResult(0)

        with self.assertRaisesRegex(ValueError, "video_path"):
            run_mlx_bridge_job(**self.bridge_kwargs(subprocess_runner=runner, mode="generate", job_id="escape"))


if __name__ == "__main__":
    unittest.main()
