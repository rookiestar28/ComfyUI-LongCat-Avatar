from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, TextIO

from .mlx_runner_contract import (
    MLX_RUNNER_SCHEMA_VERSION,
    MlxRunnerError,
    MlxRunnerRequest,
    MlxRunnerResponse,
    dump_mlx_runner_response_json,
    load_mlx_runner_request_json,
    safe_runner_log_summary,
    sanitize_log_text,
)
from .mlx_runner_validation import (
    MlxRunnerEnvironmentReport,
    MlxWeightValidationReport,
    probe_mlx_runner_environment,
    validate_mlx_weights_root,
)


@dataclass(frozen=True)
class MlxRunnerOptions:
    request_path: str
    response_path: str
    mode: str = "generate"
    probe_environment: bool = False
    timeout_seconds: float = 20.0
    log_summary: bool = False


@dataclass(frozen=True)
class MlxRunnerArtifacts:
    video_path: str = ""
    frames_path: str = ""
    timings: Mapping[str, float] = field(default_factory=dict)
    runtime: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


GenerationBackend = Callable[
    [MlxRunnerRequest, MlxWeightValidationReport, MlxRunnerEnvironmentReport | None],
    MlxRunnerArtifacts | MlxRunnerResponse | Mapping[str, Any],
]


class MlxRunnerCliError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str = "unknown",
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.diagnostics = dict(diagnostics or {})


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_response_path(response_path: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> str:
    response = Path(response_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if not _is_relative_to(response, output_root):
        raise MlxRunnerCliError(
            "response_path must stay inside output_dir.",
            stage="schema",
            diagnostics={"response_name": response.name},
        )
    return os.fspath(response)


def _write_response(response: MlxRunnerResponse, response_path: str | os.PathLike[str]) -> None:
    path = Path(response_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_mlx_runner_response_json(response, path)


def _write_error_response(
    response_path: str | os.PathLike[str],
    exc: BaseException,
    *,
    stage: str,
    diagnostics: Mapping[str, Any] | None = None,
) -> None:
    error = MlxRunnerError.from_mapping(
        {
            "error_type": exc.__class__.__name__,
            "message": sanitize_log_text(str(exc)),
            "stage": stage,
            "diagnostics": diagnostics or {},
        }
    )
    _write_response(
        MlxRunnerResponse(
            schema_version=MLX_RUNNER_SCHEMA_VERSION,
            status="error",
            error=error,
        ),
        response_path,
    )


def _runtime_summary(*, mode: str) -> dict[str, str]:
    return {
        "mode": sanitize_log_text(mode),
        "python": sanitize_log_text(platform.python_version()),
        "platform": sanitize_log_text(f"{platform.system()} {platform.machine()}"),
    }


def _safe_issues(issues: tuple[str, ...], *, limit: int = 5) -> dict[str, Any]:
    return {
        "issue_count": len(issues),
        "issues": [sanitize_log_text(issue) for issue in issues[:limit]],
    }


def _raise_if_weights_invalid(report: MlxWeightValidationReport) -> None:
    if report.is_complete:
        return
    raise MlxRunnerCliError(
        "MLX weight validation failed.",
        stage="probe",
        diagnostics={"variant": report.variant, **_safe_issues(report.issues)},
    )


def _raise_if_environment_invalid(report: MlxRunnerEnvironmentReport | None) -> None:
    if report is None or not report.issues:
        return
    raise MlxRunnerCliError(
        "MLX runner environment validation failed.",
        stage="probe",
        diagnostics={"support_status": report.support_status, **_safe_issues(report.issues)},
    )


def _default_generation_backend(
    request: MlxRunnerRequest,
    weights: MlxWeightValidationReport,
    environment: MlxRunnerEnvironmentReport | None,
) -> MlxRunnerArtifacts:
    from .mlx_generation_backend import run_mlx_generation

    return _coerce_artifacts(run_mlx_generation(request, variant_dir=weights.variant_dir))


def _coerce_artifacts(value: MlxRunnerArtifacts | MlxRunnerResponse | Mapping[str, Any]) -> MlxRunnerArtifacts:
    if isinstance(value, MlxRunnerArtifacts):
        return value
    if isinstance(value, MlxRunnerResponse):
        return MlxRunnerArtifacts(
            video_path=value.video_path,
            frames_path=value.frames_path,
            timings=value.timings,
            runtime=value.runtime,
            warnings=value.warnings,
        )
    if not isinstance(value, Mapping):
        raise TypeError("generation backend must return artifacts, response, or mapping.")
    return MlxRunnerArtifacts(
        video_path=str(value.get("video_path") or ""),
        frames_path=str(value.get("frames_path") or ""),
        timings=value.get("timings") or {},
        runtime=value.get("runtime") or {},
        warnings=tuple(sanitize_log_text(item) for item in (value.get("warnings") or ())),
    )


def _build_success_response(
    request: MlxRunnerRequest,
    artifacts: MlxRunnerArtifacts,
    *,
    require_artifacts: bool,
) -> MlxRunnerResponse:
    response = MlxRunnerResponse.from_mapping(
        {
            "schema_version": MLX_RUNNER_SCHEMA_VERSION,
            "status": "ok",
            "variant": request.variant,
            "video_path": artifacts.video_path,
            "frames_path": artifacts.frames_path,
            "timings": dict(artifacts.timings),
            "runtime": dict(artifacts.runtime),
            "warnings": list(artifacts.warnings),
        },
        output_dir=request.output_dir,
        require_artifacts=require_artifacts,
    )
    return response


def _emit_safe_summary(
    stream: TextIO | None,
    request: MlxRunnerRequest,
    response: MlxRunnerResponse,
) -> None:
    if stream is None:
        return
    stream.write(json.dumps(safe_runner_log_summary(request, response), ensure_ascii=True, sort_keys=True))
    stream.write("\n")


def run_mlx_runner(
    options: MlxRunnerOptions,
    *,
    generation_backend: GenerationBackend | None = None,
    environment_report: MlxRunnerEnvironmentReport | None = None,
    weights_report: MlxWeightValidationReport | None = None,
    output_stream: TextIO | None = None,
) -> int:
    response_path = os.fspath(options.response_path)
    request: MlxRunnerRequest | None = None
    try:
        if options.mode not in {"dry-run", "generate"}:
            raise MlxRunnerCliError("mode must be 'dry-run' or 'generate'.", stage="schema")

        request = load_mlx_runner_request_json(options.request_path)
        response_path = _resolve_response_path(options.response_path, request.output_dir)
        weights = weights_report or validate_mlx_weights_root(request.weights_root, request.variant)
        _raise_if_weights_invalid(weights)

        environment = environment_report
        if options.probe_environment:
            environment = environment or probe_mlx_runner_environment(
                sys.executable,
                variant=request.variant,
                timeout_seconds=options.timeout_seconds,
            )
        _raise_if_environment_invalid(environment)

        if options.mode == "dry-run":
            response = _build_success_response(
                request,
                MlxRunnerArtifacts(
                    runtime=_runtime_summary(mode=options.mode),
                    warnings=("dry-run: generation not executed.",) + tuple(weights.warnings),
                ),
                require_artifacts=False,
            )
            _write_response(response, response_path)
            if options.log_summary:
                _emit_safe_summary(output_stream, request, response)
            return 0

        started = time.monotonic()
        backend = generation_backend or _default_generation_backend
        artifacts = _coerce_artifacts(backend(request, weights, environment))
        timings = dict(artifacts.timings)
        timings.setdefault("runner_seconds", max(0.0, time.monotonic() - started))
        response = _build_success_response(
            request,
            MlxRunnerArtifacts(
                video_path=artifacts.video_path,
                frames_path=artifacts.frames_path,
                timings=timings,
                runtime=artifacts.runtime or _runtime_summary(mode=options.mode),
                warnings=artifacts.warnings,
            ),
            require_artifacts=True,
        )
        _write_response(response, response_path)
        if options.log_summary:
            _emit_safe_summary(output_stream, request, response)
        return 0
    except TimeoutError as exc:
        _write_error_response(
            response_path,
            exc,
            stage="inference",
            diagnostics={"mode": options.mode},
        )
        return 1
    except MlxRunnerCliError as exc:
        _write_error_response(response_path, exc, stage=exc.stage, diagnostics=exc.diagnostics)
        return 1
    except Exception as exc:
        diagnostics = {"mode": options.mode}
        if request is not None:
            diagnostics["variant"] = request.variant
        _write_error_response(response_path, exc, stage="schema" if request is None else "unknown", diagnostics=diagnostics)
        return 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongCat Avatar MLX external runner")
    parser.add_argument("--request", required=True, help="Path to LCA-084 request JSON.")
    parser.add_argument("--response", required=True, help="Path to response JSON to write.")
    parser.add_argument("--mode", choices=("dry-run", "generate"), default="generate")
    parser.add_argument(
        "--probe-environment",
        action="store_true",
        help="Probe runner Python dependencies before dry-run/generate.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--log-summary", action="store_true", help="Emit a public-safe JSON summary to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_mlx_runner(
        MlxRunnerOptions(
            request_path=args.request,
            response_path=args.response,
            mode=args.mode,
            probe_environment=args.probe_environment,
            timeout_seconds=args.timeout_seconds,
            log_summary=args.log_summary,
        ),
        output_stream=sys.stdout,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GenerationBackend",
    "MlxRunnerArtifacts",
    "MlxRunnerCliError",
    "MlxRunnerOptions",
    "build_arg_parser",
    "main",
    "run_mlx_runner",
]
