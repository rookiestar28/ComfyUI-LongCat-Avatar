from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .mlx_runner_contract import sanitize_log_text, validate_public_safe_log_payload


MLX_SMOKE_GATE_SCHEMA_VERSION = 1
MLX_SMOKE_GATE_VARIANT = "q4-merged"
MLX_SMOKE_GATE_PROFILE = (256, 432, 29)
MLX_SMOKE_MIN_UNIFIED_MEMORY_GB = 32
MLX_SMOKE_ARTIFACT_KINDS = ("mp4", "frames")
MLX_SMOKE_GATE_ALLOWED_KEYS = {
    "schema_version",
    "status",
    "variant",
    "height",
    "width",
    "num_frames",
    "host_system",
    "host_machine",
    "unified_memory_gb",
    "response_json_valid",
    "response_status",
    "artifact_present",
    "artifact_kind",
    "memory_probe_source",
    "memory_pressure",
    "timings",
    "notes",
}


@dataclass(frozen=True)
class MlxSmokeEvidence:
    schema_version: int
    status: str
    variant: str
    height: int
    width: int
    num_frames: int
    host_system: str
    host_machine: str
    unified_memory_gb: float
    response_json_valid: bool
    response_status: str
    artifact_present: bool
    artifact_kind: str
    memory_probe_source: str = "unavailable"
    memory_pressure: Mapping[str, int] = field(default_factory=dict)
    timings: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "MlxSmokeEvidence":
        if not isinstance(mapping, Mapping):
            raise TypeError("MLX smoke evidence must be a mapping.")
        unknown = sorted(set(mapping) - MLX_SMOKE_GATE_ALLOWED_KEYS)
        if unknown:
            raise ValueError(f"MLX smoke evidence contains unsupported keys: {', '.join(unknown)}")
        optional_keys = {"memory_probe_source", "memory_pressure", "notes"}
        missing = sorted(key for key in MLX_SMOKE_GATE_ALLOWED_KEYS if key not in mapping and key not in optional_keys)
        if missing:
            raise KeyError(f"MLX smoke evidence missing required keys: {', '.join(missing)}")
        timings = mapping.get("timings") or {}
        if not isinstance(timings, Mapping):
            raise TypeError("MLX smoke evidence timings must be a mapping.")
        validate_public_safe_log_payload(timings)
        normalized_timings: dict[str, float] = {}
        for key, value in timings.items():
            try:
                normalized_value = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"MLX smoke timing '{key}' must be numeric.") from exc
            if normalized_value < 0:
                raise ValueError(f"MLX smoke timing '{key}' must be non-negative.")
            normalized_timings[str(key)] = normalized_value
        memory_pressure = mapping.get("memory_pressure") or {}
        if not isinstance(memory_pressure, Mapping):
            raise TypeError("MLX smoke evidence memory_pressure must be a mapping.")
        validate_public_safe_log_payload(memory_pressure)
        normalized_pressure: dict[str, int] = {}
        for key, value in memory_pressure.items():
            try:
                normalized_value = int(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"MLX smoke memory_pressure '{key}' must be numeric.") from exc
            if normalized_value < 0:
                raise ValueError(f"MLX smoke memory_pressure '{key}' must be non-negative.")
            normalized_pressure[str(key)] = normalized_value
        notes = tuple(sanitize_log_text(item) for item in (mapping.get("notes") or ()))
        return cls(
            schema_version=int(mapping["schema_version"]),
            status=str(mapping["status"]),
            variant=str(mapping["variant"]),
            height=int(mapping["height"]),
            width=int(mapping["width"]),
            num_frames=int(mapping["num_frames"]),
            host_system=str(mapping["host_system"]),
            host_machine=str(mapping["host_machine"]),
            unified_memory_gb=float(mapping["unified_memory_gb"]),
            response_json_valid=bool(mapping["response_json_valid"]),
            response_status=str(mapping["response_status"]),
            artifact_present=bool(mapping["artifact_present"]),
            artifact_kind=str(mapping["artifact_kind"]),
            memory_probe_source=sanitize_log_text(mapping.get("memory_probe_source") or "unavailable"),
            memory_pressure=normalized_pressure,
            timings=normalized_timings,
            notes=notes,
        )


@dataclass(frozen=True)
class MlxSmokeGateDecision:
    accepted: bool
    support_status: str
    public_support_label: str
    legacy_mps_status: str
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _is_arm64(value: str) -> bool:
    return value.strip().lower().replace("-", "_") in {"arm64", "aarch64"}


def evaluate_mlx_smoke_gate(evidence: MlxSmokeEvidence | Mapping[str, Any]) -> MlxSmokeGateDecision:
    normalized = evidence if isinstance(evidence, MlxSmokeEvidence) else MlxSmokeEvidence.from_mapping(evidence)
    issues: list[str] = []
    warnings: list[str] = []

    if normalized.schema_version != MLX_SMOKE_GATE_SCHEMA_VERSION:
        issues.append(
            f"Unsupported MLX smoke schema_version {normalized.schema_version}; "
            f"expected {MLX_SMOKE_GATE_SCHEMA_VERSION}."
        )
    if normalized.variant != MLX_SMOKE_GATE_VARIANT:
        issues.append(f"First MLX support gate must use {MLX_SMOKE_GATE_VARIANT}; got {normalized.variant}.")
    profile = (normalized.height, normalized.width, normalized.num_frames)
    if profile != MLX_SMOKE_GATE_PROFILE:
        issues.append(
            "First MLX support gate must use "
            f"{MLX_SMOKE_GATE_PROFILE[0]} x {MLX_SMOKE_GATE_PROFILE[1]} x {MLX_SMOKE_GATE_PROFILE[2]}; "
            f"got {profile[0]} x {profile[1]} x {profile[2]}."
        )
    if normalized.host_system != "Darwin" or not _is_arm64(normalized.host_machine):
        issues.append("MLX support smoke requires macOS on Apple Silicon arm64.")
    if normalized.unified_memory_gb < MLX_SMOKE_MIN_UNIFIED_MEMORY_GB:
        issues.append(
            f"MLX support smoke requires {MLX_SMOKE_MIN_UNIFIED_MEMORY_GB} GB+ unified memory; "
            f"got {normalized.unified_memory_gb:g} GB."
        )
    if normalized.status != "passed":
        issues.append(f"MLX smoke status is not passed: {normalized.status}.")
    if not normalized.response_json_valid or normalized.response_status != "ok":
        issues.append("MLX smoke requires a valid ok response JSON.")
    if not normalized.artifact_present or normalized.artifact_kind not in MLX_SMOKE_ARTIFACT_KINDS:
        issues.append("MLX smoke requires an MP4 or frame artifact.")

    if issues:
        warnings.append("Apple Silicon support wording remains blocked.")
        return MlxSmokeGateDecision(
            accepted=False,
            support_status="blocked_pending_mlx_q4_artifact",
            public_support_label="",
            legacy_mps_status="blocked",
            issues=tuple(issues),
            warnings=tuple(warnings),
        )

    return MlxSmokeGateDecision(
        accepted=True,
        support_status="accepted_mlx_external_runner",
        public_support_label="Apple Silicon MLX external runner support",
        legacy_mps_status="blocked",
        issues=(),
        warnings=(),
    )


__all__ = [
    "MLX_SMOKE_ARTIFACT_KINDS",
    "MLX_SMOKE_GATE_PROFILE",
    "MLX_SMOKE_GATE_SCHEMA_VERSION",
    "MLX_SMOKE_GATE_VARIANT",
    "MLX_SMOKE_MIN_UNIFIED_MEMORY_GB",
    "MlxSmokeEvidence",
    "MlxSmokeGateDecision",
    "evaluate_mlx_smoke_gate",
]
