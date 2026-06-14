import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from LongCat_Video.mlx_runner_contract import (
    MLX_RUNNER_SCHEMA_VERSION,
    MlxRunnerError,
    MlxRunnerResponse,
    dump_mlx_runner_response_json,
)


SCRIPT_PATH = Path("scripts/run_mlx_q4_smoke_gate.py")


def load_script_module():
    spec = importlib.util.spec_from_file_location("run_mlx_q4_smoke_gate_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MlxQ4SmokeScriptTests(unittest.TestCase):
    def setUp(self):
        self.module = load_script_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output_root = self.root / "output"
        self.output_root.mkdir()
        self.image = self.root / "image.png"
        self.audio = self.root / "audio.wav"
        self.prompt = self.root / "prompt.txt"
        self.evidence = self.root / "evidence.json"
        self.image.write_bytes(b"image")
        self.audio.write_bytes(b"audio")
        self.prompt.write_text("private prompt", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def args(self, **overrides):
        data = {
            "runner_python": "python",
            "weights_root": str(self.root / "weights"),
            "image": str(self.image),
            "audio": str(self.audio),
            "prompt_file": str(self.prompt),
            "negative_prompt_file": None,
            "output_root": str(self.output_root),
            "evidence_json": str(self.evidence),
            "output_basename": "smoke",
            "fps": 30,
            "seed": 0,
            "timeout_seconds": 1.0,
            "unified_memory_gb": 32,
        }
        data.update(overrides)
        return argparse.Namespace(**data)

    def test_smoke_script_writes_accepted_content_free_evidence(self):
        def bridge_runner(**kwargs):
            job_dir = self.output_root / "job"
            job_dir.mkdir()
            frames_path = job_dir / "smoke.npy"
            response_path = job_dir / "response.json"
            frames_path.write_bytes(b"frames")
            dump_mlx_runner_response_json(
                MlxRunnerResponse(
                    schema_version=MLX_RUNNER_SCHEMA_VERSION,
                    status="ok",
                    variant="q4-merged",
                    frames_path=str(frames_path),
                    timings={"inference_seconds": 1.0},
                ),
                response_path,
            )
            return self.module.MlxBridgeResult(
                video_path="",
                frames_path=str(frames_path),
                response_path=str(response_path),
                request_path=str(job_dir / "request.json"),
                job_dir=str(job_dir),
            )

        code = self.module.run_smoke_gate(
            self.args(),
            bridge_runner=bridge_runner,
            memory_reader=lambda _: 32,
            host_reader=lambda: ("Darwin", "arm64"),
        )

        self.assertEqual(code, 0)
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(evidence["variant"], "q4-merged")
        self.assertEqual(evidence["artifact_kind"], "frames")
        self.assertEqual(evidence["memory_probe_source"], "override")
        self.assertEqual(evidence["memory_pressure"], {})
        self.assertNotIn(str(self.image), json.dumps(evidence))
        self.assertNotIn("private prompt", json.dumps(evidence))

    def test_smoke_script_records_host_memory_probe_snapshot(self):
        def bridge_runner(**kwargs):
            job_dir = self.output_root / "job"
            job_dir.mkdir()
            frames_path = job_dir / "smoke.npy"
            response_path = job_dir / "response.json"
            frames_path.write_bytes(b"frames")
            dump_mlx_runner_response_json(
                MlxRunnerResponse(
                    schema_version=MLX_RUNNER_SCHEMA_VERSION,
                    status="ok",
                    variant="q4-merged",
                    frames_path=str(frames_path),
                    timings={"inference_seconds": 1.0},
                ),
                response_path,
            )
            return self.module.MlxBridgeResult(
                video_path="",
                frames_path=str(frames_path),
                response_path=str(response_path),
                request_path=str(job_dir / "request.json"),
                job_dir=str(job_dir),
            )

        code = self.module.run_smoke_gate(
            self.args(unified_memory_gb=None),
            bridge_runner=bridge_runner,
            memory_reader=lambda _: {
                "unified_memory_gb": 32,
                "memory_probe_source": "host_sysctl",
                "memory_pressure": {"swapouts": 7, "ignored": -1},
            },
            host_reader=lambda: ("Darwin", "arm64"),
        )

        self.assertEqual(code, 0)
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(evidence["memory_probe_source"], "host_sysctl")
        self.assertEqual(evidence["memory_pressure"], {"swapouts": 7})

    def test_smoke_script_records_failed_evidence_on_bridge_error(self):
        def bridge_runner(**kwargs):
            raise TimeoutError("local path should not be serialized")

        code = self.module.run_smoke_gate(self.args(), bridge_runner=bridge_runner, memory_reader=lambda _: 32)

        self.assertEqual(code, 1)
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "failed")
        self.assertFalse(evidence["artifact_present"])
        self.assertNotIn("local path should not be serialized", json.dumps(evidence))

    def test_smoke_script_reads_error_response_after_bridge_error(self):
        def bridge_runner(**kwargs):
            job_dir = Path(kwargs["output_root"]) / f'{kwargs["output_basename"]}_{kwargs["job_id"]}'
            job_dir.mkdir()
            response_path = job_dir / "response.json"
            dump_mlx_runner_response_json(
                MlxRunnerResponse(
                    schema_version=MLX_RUNNER_SCHEMA_VERSION,
                    status="error",
                    error=MlxRunnerError(
                        error_type="MlxRunnerCliError",
                        message="MLX weight validation failed.",
                        stage="probe",
                    ),
                ),
                response_path,
            )
            raise RuntimeError("do not serialize this message")

        code = self.module.run_smoke_gate(self.args(), bridge_runner=bridge_runner, memory_reader=lambda _: 32)

        self.assertEqual(code, 1)
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertTrue(evidence["response_json_valid"])
        self.assertEqual(evidence["response_status"], "error")
        self.assertNotIn("do not serialize this message", json.dumps(evidence))

    def test_script_can_run_directly_from_repo_root(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("LongCat MLX q4 support-gate smoke", completed.stdout)


if __name__ == "__main__":
    unittest.main()
