from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import uuid
from typing import Any

from .mlx_runner_contract import (
    MLX_RUNNER_SCHEMA_VERSION,
    MlxRunnerRequest,
    dump_mlx_runner_request_json,
    load_mlx_runner_response_json,
    sanitize_log_text,
)


_SAFE_BASENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_RUNNER_LOG_TAIL_LINES = 80
_RUNNER_TERMINATE_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class MlxBridgeSubprocessResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class MlxBridgeResult:
    video_path: str
    frames_path: str
    response_path: str
    request_path: str
    job_dir: str
    log_path: str = ""
    warnings: tuple[str, ...] = ()


ImageWriter = Callable[[Any, str], None]
AudioWriter = Callable[[Any, str], None]
SubprocessRunner = Callable[[list[str], float], MlxBridgeSubprocessResult]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_basename(value: str) -> str:
    normalized = (value or "longcat_mlx").strip()
    if Path(normalized).name != normalized or normalized in {".", ".."}:
        raise ValueError("output_basename must be a filename stem, not a path.")
    if not _SAFE_BASENAME_RE.match(normalized):
        raise ValueError("output_basename may contain only letters, numbers, underscore, dash, and dot.")
    return normalized


def _resolve_output_root(output_root: str | os.PathLike[str]) -> Path:
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("ComfyUI output directory does not exist.")
    return root


def _create_job_dir(
    output_root: str | os.PathLike[str],
    output_basename: str,
    *,
    job_id: str | None = None,
) -> Path:
    root = _resolve_output_root(output_root)
    basename = _validate_basename(output_basename)
    suffix = _validate_basename(job_id or uuid.uuid4().hex[:12])
    job_dir = (root / f"{basename}_{suffix}").resolve()
    if not _is_relative_to(job_dir, root):
        raise ValueError("MLX job directory must stay under ComfyUI output directory.")
    job_dir.mkdir(parents=False, exist_ok=False)
    return job_dir


def _tail_text(lines: deque[str]) -> str:
    return "\n".join(lines)


def _coerce_timeout_output(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def _stream_pipe_to_log(
    pipe: Any,
    log_path: Path,
    tail_lines: deque[str],
    tail_lock: threading.Lock,
) -> None:
    with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
        for raw_line in iter(pipe.readline, ""):
            sanitized = sanitize_log_text(raw_line.rstrip("\r\n"))
            with tail_lock:
                tail_lines.append(sanitized)
            log_file.write(sanitized + "\n")
            log_file.flush()


def _close_pipe(pipe: Any) -> None:
    try:
        pipe.close()
    except Exception:
        pass


def _run_streaming_subprocess(
    args: list[str],
    timeout_seconds: float,
    log_path: Path,
) -> MlxBridgeSubprocessResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    tail_lines: deque[str] = deque(maxlen=_RUNNER_LOG_TAIL_LINES)
    tail_lock = threading.Lock()
    process = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("MLX runner stdout pipe was not created.")
    reader = threading.Thread(
        target=_stream_pipe_to_log,
        args=(process.stdout, log_path, tail_lines, tail_lock),
        daemon=True,
    )
    reader.start()
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        process.terminate()
        try:
            process.wait(timeout=_RUNNER_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=_RUNNER_TERMINATE_GRACE_SECONDS)
        with tail_lock:
            output = _tail_text(tail_lines)
        _close_pipe(process.stdout)
        raise subprocess.TimeoutExpired(args, timeout_seconds, output=output, stderr=output) from exc

    reader.join(timeout=_RUNNER_TERMINATE_GRACE_SECONDS)
    _close_pipe(process.stdout)
    with tail_lock:
        output = _tail_text(tail_lines)
    return MlxBridgeSubprocessResult(returncode=returncode, stdout=output, stderr=output)


def _run_subprocess(
    args: list[str],
    timeout_seconds: float,
    *,
    log_path: Path | None = None,
) -> MlxBridgeSubprocessResult:
    if log_path is not None:
        return _run_streaming_subprocess(args, timeout_seconds, log_path)
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    return MlxBridgeSubprocessResult(
        returncode=completed.returncode,
        stdout=sanitize_log_text(completed.stdout),
        stderr=sanitize_log_text(completed.stderr),
    )


def _cleanup_job_dir(job_dir: Path | None, output_root: Path, retain_job_dir: bool) -> None:
    if retain_job_dir or job_dir is None:
        return
    resolved = job_dir.resolve()
    if _is_relative_to(resolved, output_root.resolve()) and resolved.is_dir():
        shutil.rmtree(resolved)


def _write_inputs(
    *,
    image: Any,
    audio: Any,
    image_path: Path,
    audio_path: Path,
    image_writer: ImageWriter,
    audio_writer: AudioWriter,
) -> None:
    image_writer(image, os.fspath(image_path))
    audio_writer(audio, os.fspath(audio_path))
    if not image_path.is_file():
        raise FileNotFoundError("MLX bridge image writer did not create input.png.")
    if not audio_path.is_file():
        raise FileNotFoundError("MLX bridge audio writer did not create input.wav.")


def _build_subprocess_args(
    runner_python: str | os.PathLike[str],
    request_path: Path,
    response_path: Path,
    mode: str,
) -> list[str]:
    return [
        os.fspath(runner_python),
        "-u",
        "-m",
        "LongCat_Video.mlx_runner_cli",
        "--request",
        os.fspath(request_path),
        "--response",
        os.fspath(response_path),
        "--mode",
        mode,
    ]


def run_mlx_bridge_job(
    *,
    runner_python: str | os.PathLike[str],
    weights_root: str | os.PathLike[str],
    variant: str,
    image: Any,
    audio: Any,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    fps: int,
    seed: int,
    output_root: str | os.PathLike[str],
    output_basename: str = "longcat_mlx",
    mode: str = "generate",
    timeout_seconds: float = 600.0,
    retain_job_dir: bool = True,
    image_writer: ImageWriter | None = None,
    audio_writer: AudioWriter | None = None,
    subprocess_runner: SubprocessRunner | None = None,
    job_id: str | None = None,
) -> MlxBridgeResult:
    if image_writer is None:
        raise TypeError("image_writer is required.")
    if audio_writer is None:
        raise TypeError("audio_writer is required.")
    if mode not in {"dry-run", "generate"}:
        raise ValueError("mode must be 'dry-run' or 'generate'.")

    output_dir = _resolve_output_root(output_root)
    job_dir: Path | None = None
    try:
        job_dir = _create_job_dir(output_dir, output_basename, job_id=job_id)
        image_path = job_dir / "input.png"
        audio_path = job_dir / "input.wav"
        request_path = job_dir / "request.json"
        response_path = job_dir / "response.json"
        log_path = job_dir / "runner.log"

        _write_inputs(
            image=image,
            audio=audio,
            image_path=image_path,
            audio_path=audio_path,
            image_writer=image_writer,
            audio_writer=audio_writer,
        )
        request = MlxRunnerRequest.from_mapping(
            {
                "schema_version": MLX_RUNNER_SCHEMA_VERSION,
                "variant": variant,
                "weights_root": os.fspath(weights_root),
                "image_path": os.fspath(image_path),
                "audio_path": os.fspath(audio_path),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "fps": fps,
                "seed": seed,
                "output_dir": os.fspath(job_dir),
                "output_basename": _validate_basename(output_basename),
            }
        )
        dump_mlx_runner_request_json(request, request_path)

        args = _build_subprocess_args(runner_python, request_path, response_path, mode)
        try:
            if subprocess_runner is None:
                completed = _run_subprocess(args, float(timeout_seconds), log_path=log_path)
            else:
                completed = subprocess_runner(args, float(timeout_seconds))
        except subprocess.TimeoutExpired as exc:
            output = sanitize_log_text(
                _coerce_timeout_output(getattr(exc, "stderr", None))
                or _coerce_timeout_output(getattr(exc, "output", None))
            )
            message = "MLX runner timed out before writing a valid response."
            if output:
                message += f" Last runner output: {output}"
            raise TimeoutError(message) from exc

        if not response_path.is_file():
            stderr = sanitize_log_text(getattr(completed, "stderr", ""))
            if completed.returncode:
                raise RuntimeError(
                    "MLX runner exited without response JSON"
                    + (f": {stderr}" if stderr else f" (exit {completed.returncode})")
                )
            raise FileNotFoundError("MLX runner did not write response.json.")

        response = load_mlx_runner_response_json(
            response_path,
            output_dir=job_dir,
            require_artifacts=mode != "dry-run",
        )
        if response.status == "error":
            stage = response.error.stage if response.error is not None else "unknown"
            message = response.error.message if response.error is not None else "unknown runner error"
            raise RuntimeError(f"MLX runner error at {stage}: {message}")
        if completed.returncode:
            raise RuntimeError(f"MLX runner exited with code {completed.returncode} after writing success JSON.")

        return MlxBridgeResult(
            video_path=response.video_path,
            frames_path=response.frames_path,
            response_path=os.fspath(response_path),
            request_path=os.fspath(request_path),
            job_dir=os.fspath(job_dir),
            log_path=os.fspath(log_path) if log_path.is_file() else "",
            warnings=tuple(response.warnings),
        )
    except Exception:
        _cleanup_job_dir(job_dir, output_dir, retain_job_dir)
        raise


__all__ = [
    "AudioWriter",
    "ImageWriter",
    "MlxBridgeResult",
    "MlxBridgeSubprocessResult",
    "SubprocessRunner",
    "run_mlx_bridge_job",
]
