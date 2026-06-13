from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from LongCat_Video.mlx_bridge import MlxBridgeResult, run_mlx_bridge_job
from LongCat_Video.mlx_runner_contract import load_mlx_runner_response_json, sanitize_log_text
from LongCat_Video.mlx_smoke_gate import (
    MLX_SMOKE_GATE_SCHEMA_VERSION,
    evaluate_mlx_smoke_gate,
)


def _read_prompt_file(path: str | os.PathLike[str] | None) -> str:
    if path is None:
        return ""
    return Path(path).read_text(encoding="utf-8")


def _copy_file_writer(source_path: str | os.PathLike[str]):
    source = Path(source_path).expanduser().resolve()

    def writer(_payload: object, target_path: str) -> None:
        shutil.copyfile(source, target_path)

    return writer


def _host_unified_memory_gb(override: float | None = None) -> float:
    if override is not None:
        return float(override)
    if platform.system() != "Darwin":
        return 0.0
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        return int(completed.stdout.strip()) / 1024**3
    except Exception:
        return 0.0


def _response_fields(response_path: str, output_dir: str) -> tuple[bool, str, dict[str, float]]:
    try:
        response = load_mlx_runner_response_json(response_path, output_dir=output_dir, require_artifacts=False)
    except Exception:
        return False, "invalid", {}
    timings = {
        str(key): float(value)
        for key, value in response.timings.items()
        if isinstance(value, (int, float)) and float(value) >= 0
    }
    return True, response.status, timings


def _artifact_fields(result: MlxBridgeResult | None) -> tuple[bool, str]:
    if result is None:
        return False, "none"
    if result.video_path and Path(result.video_path).is_file():
        return True, "mp4"
    if result.frames_path and Path(result.frames_path).is_file():
        return True, "frames"
    return False, "none"


def _write_evidence(path: str | os.PathLike[str], evidence: dict[str, Any]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def run_smoke_gate(
    args: argparse.Namespace,
    *,
    bridge_runner=run_mlx_bridge_job,
    memory_reader=_host_unified_memory_gb,
    host_reader=lambda: (platform.system(), platform.machine()),
) -> int:
    result: MlxBridgeResult | None = None
    response_valid = False
    response_status = "missing"
    timings: dict[str, float] = {}
    notes: list[str] = []
    job_id = f"gate_{uuid.uuid4().hex[:12]}"
    expected_job_dir = Path(args.output_root).expanduser().resolve() / f"{args.output_basename}_{job_id}"
    expected_response_path = expected_job_dir / "response.json"

    try:
        result = bridge_runner(
            runner_python=args.runner_python,
            weights_root=args.weights_root,
            variant="q4-merged",
            image=object(),
            audio=object(),
            prompt=_read_prompt_file(args.prompt_file),
            negative_prompt=_read_prompt_file(args.negative_prompt_file),
            height=256,
            width=432,
            num_frames=29,
            fps=args.fps,
            seed=args.seed,
            output_root=args.output_root,
            output_basename=args.output_basename,
            mode="generate",
            timeout_seconds=args.timeout_seconds,
            retain_job_dir=True,
            job_id=job_id,
            image_writer=_copy_file_writer(args.image),
            audio_writer=_copy_file_writer(args.audio),
        )
        response_valid, response_status, timings = _response_fields(result.response_path, result.job_dir)
    except Exception as exc:
        if expected_response_path.is_file():
            response_valid, response_status, timings = _response_fields(
                os.fspath(expected_response_path),
                os.fspath(expected_job_dir),
            )
        notes.append(f"bridge error: {sanitize_log_text(exc.__class__.__name__)}")

    artifact_present, artifact_kind = _artifact_fields(result)
    host_system, host_machine = host_reader()
    evidence = {
        "schema_version": MLX_SMOKE_GATE_SCHEMA_VERSION,
        "status": "passed" if response_valid and response_status == "ok" and artifact_present else "failed",
        "variant": "q4-merged",
        "height": 256,
        "width": 432,
        "num_frames": 29,
        "host_system": str(host_system),
        "host_machine": str(host_machine),
        "unified_memory_gb": memory_reader(args.unified_memory_gb),
        "response_json_valid": response_valid,
        "response_status": response_status,
        "artifact_present": artifact_present,
        "artifact_kind": artifact_kind,
        "timings": timings,
        "notes": notes,
    }
    _write_evidence(args.evidence_json, evidence)
    decision = evaluate_mlx_smoke_gate(evidence)
    print(json.dumps({"accepted": decision.accepted, "support_status": decision.support_status}, sort_keys=True))
    return 0 if decision.accepted else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LongCat MLX q4 support-gate smoke.")
    parser.add_argument("--runner-python", required=True)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--image", required=True, help="Local reference image path. Path is not written to evidence.")
    parser.add_argument("--audio", required=True, help="Local speech audio path. Path is not written to evidence.")
    parser.add_argument("--prompt-file", required=True, help="Prompt text file. Raw prompt is not written to evidence.")
    parser.add_argument("--negative-prompt-file")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--evidence-json", required=True)
    parser.add_argument("--output-basename", default="longcat_mlx_q4_smoke")
    parser.add_argument("--fps", type=int, choices=(25, 30), default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--unified-memory-gb", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    return run_smoke_gate(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
