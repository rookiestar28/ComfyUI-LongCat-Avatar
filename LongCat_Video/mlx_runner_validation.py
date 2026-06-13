from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from .mlx_runner_contract import SUPPORTED_MLX_VARIANTS, sanitize_log_text, validate_mlx_variant


MLX_VARIANT_DIRNAMES = {
    "merged": "LongCat-Video-Avatar-1.5-bf16-dmd-merged",
    "q4-merged": "LongCat-Video-Avatar-1.5-q4-dmd-merged",
    "q8-merged": "LongCat-Video-Avatar-1.5-q8-dmd-merged",
}
MLX_VARIANT_MIN_UNIFIED_MEMORY_GB = {
    "merged": 64,
    "q4-merged": 32,
    "q8-merged": 32,
}
_EXPECTED_QUANTIZATION_BITS = {"q4-merged": 4, "q8-merged": 8}
_BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class MlxRunnerDependencyRequirement:
    import_name: str
    package_name: str
    purpose: str


MLX_RUNNER_DEPENDENCIES = (
    MlxRunnerDependencyRequirement("mlx", "mlx", "Apple MLX runtime"),
    MlxRunnerDependencyRequirement("safetensors", "safetensors", "weight file reader"),
    MlxRunnerDependencyRequirement("huggingface_hub", "huggingface-hub", "optional model download"),
    MlxRunnerDependencyRequirement("numpy", "numpy", "array preprocessing"),
    MlxRunnerDependencyRequirement("librosa", "librosa", "audio loading"),
    MlxRunnerDependencyRequirement("PIL", "Pillow", "image loading"),
    MlxRunnerDependencyRequirement("imageio", "imageio", "video export"),
    MlxRunnerDependencyRequirement("imageio_ffmpeg", "imageio-ffmpeg", "MP4 export backend"),
    MlxRunnerDependencyRequirement("transformers", "transformers", "Whisper and umT5 tokenizers"),
    MlxRunnerDependencyRequirement("mlx_arsenal", "mlx-arsenal", "FlowMatch scheduler"),
    MlxRunnerDependencyRequirement("longcat_video_avatar", "longcat-video-avatar-mlx", "MLX LongCat pipeline"),
)

_REQUIRED_COMPONENT_FILES = {
    "vae": ("config.json", "diffusion_pytorch_model.safetensors"),
    "text_encoder": ("config.json", "model.safetensors.index.json"),
    "audio_encoder": ("config.json", "model.safetensors"),
    "dit": ("config.json", "diffusion_pytorch_model.safetensors.index.json"),
    "scheduler": ("scheduler_config.json",),
    "tokenizer": ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"),
}
_SHARDED_COMPONENT_INDEXES = {
    "text_encoder": "model.safetensors.index.json",
    "dit": "diffusion_pytorch_model.safetensors.index.json",
}


@dataclass(frozen=True)
class MlxRunnerDependencyReport:
    import_name: str
    package_name: str
    purpose: str
    available: bool
    version: str = ""
    error: str = ""


@dataclass(frozen=True)
class MlxRunnerEnvironmentReport:
    runner_python: str
    executable_exists: bool
    python_executable: str
    python_version: str
    platform_system: str
    platform_machine: str
    is_macos: bool
    is_arm64: bool
    unified_memory_bytes: int | None
    dependency_reports: tuple[MlxRunnerDependencyReport, ...]
    support_status: str
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def mlx_available(self) -> bool:
        return any(item.import_name == "mlx" and item.available for item in self.dependency_reports)

    @property
    def is_generation_candidate(self) -> bool:
        return self.support_status == "generation_candidate" and not self.issues


@dataclass(frozen=True)
class MlxWeightComponentReport:
    component: str
    required_files: tuple[str, ...]
    shard_files: tuple[str, ...] = ()
    missing_files: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.missing_files and not self.issues


@dataclass(frozen=True)
class MlxWeightValidationReport:
    variant: str
    weights_root: str
    variant_dir: str
    root_mode: str
    component_reports: tuple[MlxWeightComponentReport, ...]
    quantization: Mapping[str, Any] = field(default_factory=dict)
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.issues and all(component.is_complete for component in self.component_reports)


@dataclass(frozen=True)
class MlxPreflightReport:
    environment: MlxRunnerEnvironmentReport
    weights: MlxWeightValidationReport

    @property
    def issues(self) -> tuple[str, ...]:
        return self.environment.issues + self.weights.issues

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.environment.warnings + self.weights.warnings

    @property
    def is_ready_for_generation(self) -> bool:
        return self.environment.is_generation_candidate and self.weights.is_complete


RunnerProbe = Callable[[str, str, float], str]


def _safe_name(path: str | os.PathLike[str] | None) -> str:
    if path is None:
        return "<not selected>"
    text = os.fspath(path)
    if not text:
        return "<not selected>"
    normalized = os.path.normpath(text)
    return os.path.basename(normalized) or normalized


def _coerce_int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_machine(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def classify_mlx_generation_support(
    *,
    variant: str,
    platform_system: str,
    platform_machine: str,
    unified_memory_bytes: int | None,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    normalized_variant = validate_mlx_variant(variant)
    issues: list[str] = []
    warnings: list[str] = []
    is_macos = platform_system == "Darwin"
    is_arm64 = _normalize_machine(platform_machine) in {"arm64", "aarch64"}
    if not is_macos or not is_arm64:
        issues.append("MLX generation requires macOS on Apple Silicon arm64.")
        return "unsupported_platform", tuple(issues), tuple(warnings)

    if unified_memory_bytes is None:
        warnings.append("Unified memory was not reported; generation support remains probe-only.")
        return "probe_only", tuple(issues), tuple(warnings)

    unified_memory_gb = unified_memory_bytes / _BYTES_PER_GIB
    minimum_gb = MLX_VARIANT_MIN_UNIFIED_MEMORY_GB[normalized_variant]
    if unified_memory_gb <= 16:
        warnings.append(
            "16 GB Apple Silicon hosts are install/probe-only for LongCat MLX generation "
            "until an accepted smoke artifact proves otherwise."
        )
        return "install_probe_only", tuple(issues), tuple(warnings)
    if unified_memory_gb < minimum_gb:
        warnings.append(
            f"{normalized_variant} requires {minimum_gb} GB+ unified memory for a generation candidate; "
            f"detected approximately {unified_memory_gb:.1f} GB."
        )
        return "install_probe_only", tuple(issues), tuple(warnings)
    return "generation_candidate", tuple(issues), tuple(warnings)


def build_mlx_environment_report(
    probe_data: Mapping[str, Any],
    *,
    runner_python: str | os.PathLike[str],
    variant: str = "q4-merged",
    unified_memory_bytes: int | None = None,
) -> MlxRunnerEnvironmentReport:
    if not isinstance(probe_data, Mapping):
        raise TypeError("MLX runner environment probe data must be a mapping.")
    normalized_variant = validate_mlx_variant(variant)
    runner_path = Path(runner_python).expanduser()
    dependency_data = probe_data.get("dependencies") or {}
    if not isinstance(dependency_data, Mapping):
        raise TypeError("MLX runner environment dependency data must be a mapping.")

    dependency_reports: list[MlxRunnerDependencyReport] = []
    issues: list[str] = []
    warnings: list[str] = []
    for requirement in MLX_RUNNER_DEPENDENCIES:
        item = dependency_data.get(requirement.import_name) or {}
        if not isinstance(item, Mapping):
            item = {}
        available = bool(item.get("available"))
        version = sanitize_log_text(item.get("version") or "")
        error = sanitize_log_text(item.get("error") or "")
        dependency_reports.append(
            MlxRunnerDependencyReport(
                import_name=requirement.import_name,
                package_name=requirement.package_name,
                purpose=requirement.purpose,
                available=available,
                version=version,
                error=error,
            )
        )
        if not available:
            detail = f" ({error})" if error else ""
            issues.append(
                f"Install missing MLX runner dependency '{requirement.package_name}' "
                f"for {requirement.purpose}.{detail}"
            )

    executable_exists = bool(probe_data.get("executable_exists", runner_path.is_file()))
    if not executable_exists:
        issues.append(f"Runner Python does not exist or is not a file: {_safe_name(runner_path)}.")

    detected_memory = _coerce_int_or_none(probe_data.get("unified_memory_bytes"))
    if unified_memory_bytes is not None:
        detected_memory = unified_memory_bytes
    platform_system = str(probe_data.get("platform_system") or "")
    platform_machine = str(probe_data.get("platform_machine") or "")
    support_status, support_issues, support_warnings = classify_mlx_generation_support(
        variant=normalized_variant,
        platform_system=platform_system,
        platform_machine=platform_machine,
        unified_memory_bytes=detected_memory,
    )
    issues.extend(support_issues)
    warnings.extend(support_warnings)

    return MlxRunnerEnvironmentReport(
        runner_python=os.fspath(runner_path),
        executable_exists=executable_exists,
        python_executable=sanitize_log_text(probe_data.get("python_executable") or ""),
        python_version=sanitize_log_text(probe_data.get("python_version") or ""),
        platform_system=platform_system,
        platform_machine=platform_machine,
        is_macos=platform_system == "Darwin",
        is_arm64=_normalize_machine(platform_machine) in {"arm64", "aarch64"},
        unified_memory_bytes=detected_memory,
        dependency_reports=tuple(dependency_reports),
        support_status=support_status,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def _build_runner_probe_script() -> str:
    requirements = [
        {
            "import_name": item.import_name,
            "package_name": item.package_name,
            "purpose": item.purpose,
        }
        for item in MLX_RUNNER_DEPENDENCIES
    ]
    # CRITICAL: keep runner-only dependency imports inside this subprocess
    # script; importing mlx in this module breaks CI and default ComfyUI envs.
    return f"""
import importlib
import importlib.metadata
import json
import platform
import sys

requirements = {requirements!r}
dependencies = {{}}
for requirement in requirements:
    import_name = requirement["import_name"]
    package_name = requirement["package_name"]
    try:
        importlib.import_module(import_name)
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = ""
        dependencies[import_name] = {{
            "available": True,
            "version": version,
            "error": "",
        }}
    except Exception as exc:
        dependencies[import_name] = {{
            "available": False,
            "version": "",
            "error": exc.__class__.__name__,
        }}

print(json.dumps({{
    "executable_exists": True,
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "platform_system": platform.system(),
    "platform_machine": platform.machine(),
    "dependencies": dependencies,
}}, sort_keys=True))
"""


def _run_probe_subprocess(runner_python: str, script: str, timeout_seconds: float) -> str:
    completed = subprocess.run(
        [runner_python, "-c", script],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        stderr = sanitize_log_text(completed.stderr.strip())
        raise RuntimeError(
            "MLX runner environment probe failed"
            + (f": {stderr}" if stderr else f" with exit code {completed.returncode}")
        )
    return completed.stdout


def probe_mlx_runner_environment(
    runner_python: str | os.PathLike[str],
    *,
    variant: str = "q4-merged",
    timeout_seconds: float = 20.0,
    run_probe: RunnerProbe | None = None,
    unified_memory_bytes: int | None = None,
) -> MlxRunnerEnvironmentReport:
    runner_path = Path(runner_python).expanduser()
    if not runner_path.is_file():
        return build_mlx_environment_report(
            {
                "executable_exists": False,
                "platform_system": platform.system(),
                "platform_machine": platform.machine(),
                "dependencies": {},
            },
            runner_python=runner_path,
            variant=variant,
            unified_memory_bytes=unified_memory_bytes,
        )

    script = _build_runner_probe_script()
    output = (run_probe or _run_probe_subprocess)(os.fspath(runner_path), script, timeout_seconds)
    try:
        probe_data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed MLX runner environment probe JSON.") from exc
    return build_mlx_environment_report(
        probe_data,
        runner_python=runner_path,
        variant=variant,
        unified_memory_bytes=unified_memory_bytes,
    )


def _variant_dir_for_root(root: Path, variant: str) -> tuple[Path, str, tuple[str, ...]]:
    dirname = MLX_VARIANT_DIRNAMES[variant]
    nested = root / dirname
    if nested.is_dir():
        return nested, "weights_root_parent", ()
    if root.name == dirname and root.is_dir():
        return root, "variant_dir", ()
    return nested, "missing", (f"Missing MLX variant directory: {dirname}.",)


def _read_json_file(path: Path, label: str) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, (f"Missing {label}: {path.name}.",)
    except json.JSONDecodeError:
        return None, (f"Malformed {label}: {path.name}.",)
    if not isinstance(data, Mapping):
        return None, (f"Malformed {label}: expected JSON object.",)
    return data, ()


def _validate_sharded_index(component_dir: Path, index_name: str, component: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    index_data, index_issues = _read_json_file(component_dir / index_name, f"{component} sharded index")
    if index_issues or index_data is None:
        return (), (), index_issues
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        return (), (), (f"Malformed {component} sharded index: missing non-empty weight_map.",)

    shard_files: set[str] = set()
    missing_files: set[str] = set()
    issues: list[str] = []
    component_root = component_dir.resolve()
    for shard_reference in weight_map.values():
        if not isinstance(shard_reference, str):
            issues.append(f"Unsafe {component} shard reference: filename must be a string.")
            continue
        normalized = os.path.normpath(shard_reference)
        if normalized == "." or normalized.startswith("..") or os.path.isabs(normalized):
            issues.append(f"Unsafe {component} shard reference: {shard_reference}.")
            continue
        shard_path = (component_dir / normalized).resolve()
        try:
            shard_path.relative_to(component_root)
        except ValueError:
            issues.append(f"Unsafe {component} shard reference escapes component directory.")
            continue
        if shard_path.suffix.lower() != ".safetensors":
            issues.append(f"Unsupported {component} shard extension: {Path(normalized).name}.")
            continue
        relative = f"{component}/{normalized.replace(os.sep, '/')}"
        if not shard_path.is_file():
            missing_files.add(relative)
            continue
        shard_files.add(relative)
    return tuple(sorted(shard_files)), tuple(sorted(missing_files)), tuple(issues)


def _validate_component(variant_dir: Path, component: str, required_files: tuple[str, ...]) -> MlxWeightComponentReport:
    component_dir = variant_dir / component
    required = tuple(f"{component}/{name}" for name in required_files)
    missing: list[str] = []
    issues: list[str] = []
    shard_files: tuple[str, ...] = ()
    if not component_dir.is_dir():
        return MlxWeightComponentReport(
            component=component,
            required_files=required,
            missing_files=(f"{component}/",) + required,
            issues=(f"Missing MLX component directory: {component}.",),
        )

    for filename in required_files:
        if not (component_dir / filename).is_file():
            missing.append(f"{component}/{filename}")

    index_name = _SHARDED_COMPONENT_INDEXES.get(component)
    if index_name and (component_dir / index_name).is_file():
        shard_files, shard_missing, shard_issues = _validate_sharded_index(component_dir, index_name, component)
        missing.extend(shard_missing)
        issues.extend(shard_issues)
    return MlxWeightComponentReport(
        component=component,
        required_files=required,
        shard_files=shard_files,
        missing_files=tuple(sorted(set(missing))),
        issues=tuple(issues),
    )


def _validate_dit_quantization(variant_dir: Path, variant: str) -> tuple[Mapping[str, Any], tuple[str, ...], tuple[str, ...]]:
    config_data, config_issues = _read_json_file(variant_dir / "dit" / "config.json", "DiT config")
    if config_issues or config_data is None:
        return {}, config_issues, ()
    quantization = config_data.get("quantization")
    expected_bits = _EXPECTED_QUANTIZATION_BITS.get(variant)
    if expected_bits is None:
        if quantization:
            return {}, (f"{variant} should not include DiT quantization metadata.",), ()
        return {}, (), ()
    if not isinstance(quantization, Mapping):
        return {}, (f"{variant} requires DiT quantization metadata in dit/config.json.",), ()

    issues: list[str] = []
    bits = quantization.get("bits")
    group_size = quantization.get("group_size")
    if bits != expected_bits:
        issues.append(f"{variant} expected {expected_bits}-bit DiT quantization; found {bits!r}.")
    if group_size != 64:
        issues.append(f"{variant} expected DiT quantization group_size 64; found {group_size!r}.")
    skip_patterns = quantization.get("skip_patterns")
    if skip_patterns is not None and not isinstance(skip_patterns, list):
        issues.append(f"{variant} DiT quantization skip_patterns must be a list when present.")
    return dict(quantization), tuple(issues), ()


def validate_mlx_weights_root(
    weights_root: str | os.PathLike[str],
    variant: str,
) -> MlxWeightValidationReport:
    normalized_variant = validate_mlx_variant(variant)
    if normalized_variant not in MLX_VARIANT_DIRNAMES:
        raise ValueError(f"Unsupported MLX weights variant: {normalized_variant}")

    root = Path(weights_root).expanduser().resolve()
    issues: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        return MlxWeightValidationReport(
            variant=normalized_variant,
            weights_root=os.fspath(root),
            variant_dir=os.fspath(root / MLX_VARIANT_DIRNAMES[normalized_variant]),
            root_mode="missing",
            component_reports=(),
            issues=(f"MLX weights_root does not exist or is not a directory: {_safe_name(root)}.",),
        )

    variant_dir, root_mode, variant_issues = _variant_dir_for_root(root, normalized_variant)
    issues.extend(variant_issues)
    if variant_issues:
        return MlxWeightValidationReport(
            variant=normalized_variant,
            weights_root=os.fspath(root),
            variant_dir=os.fspath(variant_dir),
            root_mode=root_mode,
            component_reports=(),
            issues=tuple(issues),
        )
    if root_mode == "variant_dir":
        warnings.append(
            "weights_root points directly at the selected variant directory; parent weights root is preferred."
        )

    component_reports = tuple(
        _validate_component(variant_dir, component, required_files)
        for component, required_files in _REQUIRED_COMPONENT_FILES.items()
    )
    for component_report in component_reports:
        for missing in component_report.missing_files:
            issues.append(f"Missing MLX weight file: {missing}.")
        issues.extend(component_report.issues)

    quantization, quantization_issues, quantization_warnings = _validate_dit_quantization(
        variant_dir,
        normalized_variant,
    )
    issues.extend(quantization_issues)
    warnings.extend(quantization_warnings)

    return MlxWeightValidationReport(
        variant=normalized_variant,
        weights_root=os.fspath(root),
        variant_dir=os.fspath(variant_dir),
        root_mode=root_mode,
        component_reports=component_reports,
        quantization=quantization,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def validate_mlx_preflight(
    *,
    runner_python: str | os.PathLike[str],
    weights_root: str | os.PathLike[str],
    variant: str,
    timeout_seconds: float = 20.0,
    run_probe: RunnerProbe | None = None,
    unified_memory_bytes: int | None = None,
) -> MlxPreflightReport:
    environment = probe_mlx_runner_environment(
        runner_python,
        variant=variant,
        timeout_seconds=timeout_seconds,
        run_probe=run_probe,
        unified_memory_bytes=unified_memory_bytes,
    )
    weights = validate_mlx_weights_root(weights_root, variant)
    return MlxPreflightReport(environment=environment, weights=weights)


__all__ = [
    "MLX_RUNNER_DEPENDENCIES",
    "MLX_VARIANT_DIRNAMES",
    "MLX_VARIANT_MIN_UNIFIED_MEMORY_GB",
    "MlxPreflightReport",
    "MlxRunnerDependencyReport",
    "MlxRunnerDependencyRequirement",
    "MlxRunnerEnvironmentReport",
    "MlxWeightComponentReport",
    "MlxWeightValidationReport",
    "build_mlx_environment_report",
    "classify_mlx_generation_support",
    "probe_mlx_runner_environment",
    "validate_mlx_preflight",
    "validate_mlx_weights_root",
]
