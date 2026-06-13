from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any


MLX_RUNNER_SCHEMA_VERSION = 1
SUPPORTED_MLX_VARIANTS = ("q4-merged", "q8-merged", "merged")

# Keep the first schema deliberately narrow until LCA-088 produces accepted
# artifact evidence for broader resolutions or continuation profiles.
SUPPORTED_MLX_GENERATION_PROFILES = ((256, 432, 29),)
SUPPORTED_MLX_FPS = (25, 30)
MAX_SEED = 2**31 - 1
MAX_PROMPT_CHARS = 8000
MAX_LOG_TEXT_CHARS = 2000

_SAFE_BASENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SENSITIVE_KEY_RE = re.compile(
    r"(token|secret|password|passwd|cookie|credential|authorization|api[_-]?key)",
    re.IGNORECASE,
)
_RAW_MEDIA_KEY_RE = re.compile(
    r"(raw[_-]?(image|audio|media)|image[_-]?bytes|audio[_-]?bytes|media[_-]?payload|base64)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE_RE = re.compile(
    r"(?i)\b(token|secret|password|passwd|cookie|authorization|api[_-]?key)\s*[:=]\s*[^,\s;]+"
)
_TOKEN_PREFIX_RE = re.compile(r"\b(?:hf_|ghp_|sk-)[A-Za-z0-9_\-]{8,}")
_INTERNAL_OUTPUT_PARTS = {"reference", ".planning", ".sessions"}

_REQUEST_KEYS = {
    "schema_version",
    "variant",
    "weights_root",
    "image_path",
    "audio_path",
    "prompt",
    "negative_prompt",
    "height",
    "width",
    "num_frames",
    "fps",
    "seed",
    "output_dir",
    "output_basename",
}
_SUCCESS_RESPONSE_KEYS = {
    "schema_version",
    "status",
    "variant",
    "video_path",
    "frames_path",
    "timings",
    "runtime",
    "warnings",
}
_ERROR_RESPONSE_KEYS = {
    "schema_version",
    "status",
    "error_type",
    "message",
    "stage",
    "diagnostics",
}
_ERROR_STAGES = ("schema", "probe", "load", "preprocess", "tokenize", "inference", "export", "dry_run", "unknown")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return os.fspath(value)
    return value


def _read_json_object(path: str | os.PathLike[str], role: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {role} JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Malformed {role} JSON: expected an object.")
    return data


def _reject_unknown_keys(mapping: Mapping[str, Any], allowed: set[str], role: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{role} contains unsupported keys: {', '.join(unknown)}")
    missing = sorted(key for key in allowed if key not in mapping)
    if missing:
        raise KeyError(f"{role} missing required keys: {', '.join(missing)}")


def _reject_sensitive_or_raw_keys(mapping: Mapping[str, Any], role: str) -> None:
    for key, value in mapping.items():
        text_key = str(key)
        if _SENSITIVE_KEY_RE.search(text_key):
            raise ValueError(f"{role} contains sensitive key '{text_key}'.")
        if _RAW_MEDIA_KEY_RE.search(text_key):
            raise ValueError(f"{role} contains raw media payload key '{text_key}'.")
        if isinstance(value, Mapping):
            _reject_sensitive_or_raw_keys(value, f"{role}.{text_key}")


def sanitize_log_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = _SENSITIVE_VALUE_RE.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = _TOKEN_PREFIX_RE.sub("<redacted>", text)
    if len(text) > MAX_LOG_TEXT_CHARS:
        text = text[:MAX_LOG_TEXT_CHARS] + "...<truncated>"
    return text


def validate_public_safe_log_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("log payload must be a mapping.")
    _reject_sensitive_or_raw_keys(payload, "log payload")


def _coerce_string(value: Any, role: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{role} must be a string.")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise ValueError(f"{role} must be non-empty.")
    return normalized


def _coerce_int(value: Any, role: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{role} must be an integer.") from exc
    return normalized


def _resolve_existing_file(value: str | os.PathLike[str], role: str) -> str:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{role} does not point to an existing file.")
    return os.fspath(path)


def _resolve_existing_dir(value: str | os.PathLike[str], role: str) -> str:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{role} does not point to an existing directory.")
    return os.fspath(path)


def _resolve_safe_output_dir(value: str | os.PathLike[str]) -> str:
    output_dir = Path(value).expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("output_dir does not point to an existing directory.")
    lowered_parts = {part.lower() for part in output_dir.parts}
    if lowered_parts & _INTERNAL_OUTPUT_PARTS:
        raise ValueError("output_dir must not be inside internal planning, reference, or session directories.")
    return os.fspath(output_dir)


def _validate_output_basename(value: str) -> str:
    basename = _coerce_string(value, "output_basename")
    if Path(basename).name != basename or basename in {".", ".."}:
        raise ValueError("output_basename must be a filename stem, not a path.")
    if not _SAFE_BASENAME_RE.match(basename):
        raise ValueError("output_basename may contain only letters, numbers, underscore, dash, and dot.")
    return basename


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_output_artifact_path(
    value: str | os.PathLike[str] | None,
    *,
    output_dir: str | os.PathLike[str],
    role: str,
    require_exists: bool,
) -> str:
    if value in (None, ""):
        return ""
    artifact = Path(value).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    if not _is_relative_to(artifact, root):
        raise ValueError(f"{role} must stay inside output_dir.")
    if require_exists and not artifact.is_file():
        raise FileNotFoundError(f"{role} does not point to an existing file.")
    return os.fspath(artifact)


def _validate_schema_version(value: Any) -> int:
    version = _coerce_int(value, "schema_version")
    if version != MLX_RUNNER_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported MLX runner schema_version {version}; expected {MLX_RUNNER_SCHEMA_VERSION}."
        )
    return version


def validate_mlx_variant(value: Any) -> str:
    variant = _coerce_string(value, "variant")
    if variant not in SUPPORTED_MLX_VARIANTS:
        raise ValueError(
            f"Unsupported MLX runner variant '{variant}'. Expected one of {SUPPORTED_MLX_VARIANTS}."
        )
    return variant


def validate_generation_controls(height: Any, width: Any, num_frames: Any, fps: Any, seed: Any) -> tuple[int, int, int, int, int]:
    normalized_height = _coerce_int(height, "height")
    normalized_width = _coerce_int(width, "width")
    normalized_frames = _coerce_int(num_frames, "num_frames")
    normalized_fps = _coerce_int(fps, "fps")
    normalized_seed = _coerce_int(seed, "seed")
    if (normalized_height, normalized_width, normalized_frames) not in SUPPORTED_MLX_GENERATION_PROFILES:
        raise ValueError(
            "Unsupported MLX generation profile "
            f"{normalized_height}x{normalized_width}x{normalized_frames}. "
            f"Supported profiles: {SUPPORTED_MLX_GENERATION_PROFILES}."
        )
    if normalized_fps not in SUPPORTED_MLX_FPS:
        raise ValueError(f"Unsupported fps {normalized_fps}. Expected one of {SUPPORTED_MLX_FPS}.")
    if normalized_seed < 0 or normalized_seed > MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}.")
    return normalized_height, normalized_width, normalized_frames, normalized_fps, normalized_seed


def _validate_prompt(value: Any, role: str, *, allow_empty: bool = False) -> str:
    prompt = _coerce_string(value, role, allow_empty=allow_empty)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"{role} exceeds {MAX_PROMPT_CHARS} characters.")
    return prompt


@dataclass(frozen=True)
class MlxRunnerRequest:
    schema_version: int
    variant: str
    weights_root: str
    image_path: str
    audio_path: str
    prompt: str
    negative_prompt: str
    height: int
    width: int
    num_frames: int
    fps: int
    seed: int
    output_dir: str
    output_basename: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "MlxRunnerRequest":
        if not isinstance(mapping, Mapping):
            raise TypeError("MLX runner request must be a mapping.")
        _reject_unknown_keys(mapping, _REQUEST_KEYS, "MLX runner request")
        _reject_sensitive_or_raw_keys(mapping, "MLX runner request")
        height, width, num_frames, fps, seed = validate_generation_controls(
            mapping["height"],
            mapping["width"],
            mapping["num_frames"],
            mapping["fps"],
            mapping["seed"],
        )
        return cls(
            schema_version=_validate_schema_version(mapping["schema_version"]),
            variant=validate_mlx_variant(mapping["variant"]),
            weights_root=_resolve_existing_dir(mapping["weights_root"], "weights_root"),
            image_path=_resolve_existing_file(mapping["image_path"], "image_path"),
            audio_path=_resolve_existing_file(mapping["audio_path"], "audio_path"),
            prompt=_validate_prompt(mapping["prompt"], "prompt"),
            negative_prompt=_validate_prompt(mapping["negative_prompt"], "negative_prompt", allow_empty=True),
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            output_dir=_resolve_safe_output_dir(mapping["output_dir"]),
            output_basename=_validate_output_basename(mapping["output_basename"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "variant": self.variant,
            "weights_root": self.weights_root,
            "image_path": self.image_path,
            "audio_path": self.audio_path,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "output_basename": self.output_basename,
        }


@dataclass(frozen=True)
class MlxRunnerError:
    error_type: str
    message: str
    stage: str = "unknown"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "MlxRunnerError":
        error_type = _coerce_string(mapping.get("error_type"), "error_type")
        message = sanitize_log_text(_coerce_string(mapping.get("message"), "message"))
        stage = _coerce_string(mapping.get("stage", "unknown"), "stage")
        if stage not in _ERROR_STAGES:
            raise ValueError(f"Unsupported error stage '{stage}'. Expected one of {_ERROR_STAGES}.")
        diagnostics = mapping.get("diagnostics") or {}
        if not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping.")
        validate_public_safe_log_payload(diagnostics)
        return cls(error_type=error_type, message=message, stage=stage, diagnostics=dict(diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "stage": self.stage,
            "diagnostics": _json_safe(self.diagnostics),
        }


@dataclass(frozen=True)
class MlxRunnerResponse:
    schema_version: int
    status: str
    variant: str | None = None
    video_path: str = ""
    frames_path: str = ""
    timings: Mapping[str, float] = field(default_factory=dict)
    runtime: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: MlxRunnerError | None = None

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        output_dir: str | os.PathLike[str] | None = None,
        require_artifacts: bool = True,
    ) -> "MlxRunnerResponse":
        if not isinstance(mapping, Mapping):
            raise TypeError("MLX runner response must be a mapping.")
        status = _coerce_string(mapping.get("status"), "status")
        if status == "ok":
            _reject_unknown_keys(mapping, _SUCCESS_RESPONSE_KEYS, "MLX runner success response")
            if output_dir is None:
                raise ValueError("output_dir is required to validate success response artifacts.")
            timings = _validate_timings(mapping.get("timings") or {})
            runtime = _validate_runtime(mapping.get("runtime") or {})
            warnings = tuple(sanitize_log_text(item) for item in (mapping.get("warnings") or ()))
            video_path = _resolve_output_artifact_path(
                mapping.get("video_path"),
                output_dir=output_dir,
                role="video_path",
                require_exists=require_artifacts,
            )
            frames_path = _resolve_output_artifact_path(
                mapping.get("frames_path"),
                output_dir=output_dir,
                role="frames_path",
                require_exists=require_artifacts,
            )
            if not video_path and not frames_path:
                raise ValueError("success response must include video_path or frames_path.")
            return cls(
                schema_version=_validate_schema_version(mapping["schema_version"]),
                status=status,
                variant=validate_mlx_variant(mapping["variant"]),
                video_path=video_path,
                frames_path=frames_path,
                timings=timings,
                runtime=runtime,
                warnings=warnings,
                error=None,
            )
        if status == "error":
            _reject_unknown_keys(mapping, _ERROR_RESPONSE_KEYS, "MLX runner error response")
            error = MlxRunnerError.from_mapping(mapping)
            return cls(
                schema_version=_validate_schema_version(mapping["schema_version"]),
                status=status,
                error=error,
            )
        raise ValueError("MLX runner response status must be 'ok' or 'error'.")

    def to_dict(self) -> dict[str, Any]:
        if self.status == "error":
            if self.error is None:
                raise ValueError("error response requires an error payload.")
            return {
                "schema_version": self.schema_version,
                "status": self.status,
                **self.error.to_dict(),
            }
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "variant": self.variant,
            "video_path": self.video_path,
            "frames_path": self.frames_path,
            "timings": _json_safe(self.timings),
            "runtime": _json_safe(self.runtime),
            "warnings": list(self.warnings),
        }


def _validate_timings(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("timings must be a mapping.")
    _reject_sensitive_or_raw_keys(value, "timings")
    normalized: dict[str, float] = {}
    for key, item in value.items():
        try:
            seconds = float(item)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"timing '{key}' must be numeric.") from exc
        if seconds < 0:
            raise ValueError(f"timing '{key}' must be non-negative.")
        normalized[str(key)] = seconds
    return normalized


def _validate_runtime(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime must be a mapping.")
    validate_public_safe_log_payload(value)
    return {str(key): sanitize_log_text(item) for key, item in value.items()}


def validate_mlx_runner_request(request: MlxRunnerRequest) -> MlxRunnerRequest:
    return MlxRunnerRequest.from_mapping(request.to_dict())


def validate_mlx_runner_response(
    response: MlxRunnerResponse,
    *,
    output_dir: str | os.PathLike[str] | None = None,
    require_artifacts: bool = True,
) -> MlxRunnerResponse:
    return MlxRunnerResponse.from_mapping(
        response.to_dict(),
        output_dir=output_dir,
        require_artifacts=require_artifacts,
    )


def load_mlx_runner_request_json(path: str | os.PathLike[str]) -> MlxRunnerRequest:
    return MlxRunnerRequest.from_mapping(_read_json_object(path, "MLX runner request"))


def load_mlx_runner_response_json(
    path: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    require_artifacts: bool = True,
) -> MlxRunnerResponse:
    return MlxRunnerResponse.from_mapping(
        _read_json_object(path, "MLX runner response"),
        output_dir=output_dir,
        require_artifacts=require_artifacts,
    )


def dump_mlx_runner_request_json(request: MlxRunnerRequest, path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(request.to_dict(), handle, ensure_ascii=True, indent=2, sort_keys=True)


def dump_mlx_runner_response_json(response: MlxRunnerResponse, path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(response.to_dict(), handle, ensure_ascii=True, indent=2, sort_keys=True)


def safe_runner_log_summary(request: MlxRunnerRequest, response: MlxRunnerResponse | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": request.schema_version,
        "variant": request.variant,
        "generation": {
            "height": request.height,
            "width": request.width,
            "num_frames": request.num_frames,
            "fps": request.fps,
            "seed": request.seed,
        },
        "inputs": {
            "weights_root": Path(request.weights_root).name,
            "image": Path(request.image_path).name,
            "audio": Path(request.audio_path).name,
        },
        "prompt": {
            "prompt_chars": len(request.prompt),
            "negative_prompt_chars": len(request.negative_prompt),
        },
        "output": {
            "directory": Path(request.output_dir).name,
            "basename": request.output_basename,
        },
    }
    if response is not None:
        summary["response"] = {"status": response.status}
        if response.status == "ok":
            summary["response"].update(
                {
                    "has_video": bool(response.video_path),
                    "has_frames": bool(response.frames_path),
                    "warnings": tuple(response.warnings),
                }
            )
        elif response.error is not None:
            summary["response"].update(
                {
                    "error_type": response.error.error_type,
                    "stage": response.error.stage,
                    "message": sanitize_log_text(response.error.message),
                }
            )
    validate_public_safe_log_payload(summary)
    return summary


__all__ = [
    "MLX_RUNNER_SCHEMA_VERSION",
    "SUPPORTED_MLX_FPS",
    "SUPPORTED_MLX_GENERATION_PROFILES",
    "SUPPORTED_MLX_VARIANTS",
    "MlxRunnerError",
    "MlxRunnerRequest",
    "MlxRunnerResponse",
    "dump_mlx_runner_request_json",
    "dump_mlx_runner_response_json",
    "load_mlx_runner_request_json",
    "load_mlx_runner_response_json",
    "safe_runner_log_summary",
    "sanitize_log_text",
    "validate_generation_controls",
    "validate_mlx_runner_request",
    "validate_mlx_runner_response",
    "validate_mlx_variant",
    "validate_public_safe_log_payload",
]
