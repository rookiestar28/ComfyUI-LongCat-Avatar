from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from LongCat_Video.backend_capabilities import (
    describe_backend,
    empty_cache,
    mps_cpu_fallback_enabled,
    read_memory_stats,
    synchronize,
)
from LongCat_Video.backend_dtype_policy import resolve_backend_dtype_policy
from LongCat_Video.mps_attention_probe import run_mps_attention_equivalence_probe


STATUS_PASS = "pass"
STATUS_BLOCKED = "blocked"
STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class SmokeStep:
    name: str
    status: str
    native_mps: bool
    cpu_fallback_enabled: bool
    detail: str


@dataclass(frozen=True)
class SmokeMatrixResult:
    status: str
    environment: dict[str, str]
    steps: tuple[SmokeStep, ...]
    blockers: tuple[str, ...]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "environment": dict(self.environment),
            "steps": [asdict(step) for step in self.steps],
            "blockers": list(self.blockers),
        }


def run_mps_smoke_matrix(
    *,
    torch_module: Any = None,
    environ: Mapping[str, str] | None = None,
    branch: str = "unknown",
    commit: str = "unknown",
) -> SmokeMatrixResult:
    environ = environ or {}
    fallback_enabled = mps_cpu_fallback_enabled(environ)
    torch_module, torch_detail = _resolve_torch(torch_module)
    environment = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch_detail,
        "branch": branch,
        "commit": commit,
        "pytorch_enable_mps_fallback": "true" if fallback_enabled else "false",
    }

    steps: list[SmokeStep] = []
    blockers: list[str] = []

    if torch_module is None:
        blockers.append("PyTorch is not installed in this environment.")
        steps.append(_blocked_step("mps_available", fallback_enabled, blockers[-1]))
        steps.extend(_blocked_component_steps(fallback_enabled, "PyTorch is required."))
        return _result(environment, steps, blockers)

    caps = describe_backend("mps", torch_module=torch_module, environ=environ)
    if fallback_enabled:
        blockers.append("PYTORCH_ENABLE_MPS_FALLBACK is enabled; native MPS evidence requires fallback disabled.")

    if not caps.available:
        blockers.append("MPS backend is not available.")
        steps.append(_blocked_step("mps_available", fallback_enabled, "MPS backend is not available."))
        steps.extend(_blocked_component_steps(fallback_enabled, "MPS backend is unavailable."))
        return _result(environment, steps, blockers)

    steps.append(SmokeStep("mps_available", STATUS_PASS, True, fallback_enabled, "MPS backend is available."))

    try:
        dtype_policy = resolve_backend_dtype_policy("mps", torch_module=torch_module)
        steps.append(
            SmokeStep(
                "dtype_policy",
                STATUS_PASS,
                True,
                fallback_enabled,
                f"text={dtype_policy.text_encoder_precision}; audio={dtype_policy.audio_encoder_precision}; "
                f"dit={dtype_policy.dit_precision}; vae={dtype_policy.vae_precision}; math={dtype_policy.math_precision}",
            )
        )
    except Exception as exc:
        blockers.append(f"dtype policy failed: {type(exc).__name__}: {exc}")
        steps.append(_blocked_step("dtype_policy", fallback_enabled, blockers[-1]))

    cache_result = empty_cache("mps", torch_module=torch_module)
    sync_result = synchronize("mps", torch_module=torch_module)
    memory_stats = read_memory_stats("mps", torch_module=torch_module)
    diagnostics_ok = cache_result.success and sync_result.success and memory_stats.available
    if not diagnostics_ok:
        blockers.append(
            "MPS diagnostics incomplete: "
            f"empty_cache={cache_result.detail}; synchronize={sync_result.detail}; memory={memory_stats.detail}"
        )
    steps.append(
        SmokeStep(
            "cache_sync_memory",
            STATUS_PASS if diagnostics_ok else STATUS_BLOCKED,
            diagnostics_ok,
            fallback_enabled,
            "ok" if diagnostics_ok else blockers[-1],
        )
    )

    attention_result = run_mps_attention_equivalence_probe(torch_module=torch_module, environ=environ)
    if attention_result.status != STATUS_PASS:
        blockers.append(f"attention probe {attention_result.status}: {attention_result.reason}")
    steps.append(
        SmokeStep(
            "attention_probe",
            attention_result.status,
            attention_result.native_mps,
            attention_result.cpu_fallback_enabled,
            attention_result.reason,
        )
    )

    steps.extend(
        _blocked_component_steps(
            fallback_enabled,
            "Real model assets and Apple Silicon execution are required for this component.",
        )
    )
    blockers.extend(step.detail for step in steps if step.status == STATUS_BLOCKED and step.name.startswith("component_"))
    return _result(environment, steps, blockers)


def _resolve_torch(torch_module: Any) -> tuple[Any, str]:
    if torch_module is not None:
        version = getattr(torch_module, "__version__", "injected")
        return torch_module, str(version)
    try:
        import torch as imported_torch
    except ModuleNotFoundError:
        return None, "unavailable"
    return imported_torch, str(getattr(imported_torch, "__version__", "unknown"))


def _blocked_step(name: str, fallback_enabled: bool, detail: str) -> SmokeStep:
    return SmokeStep(name, STATUS_BLOCKED, False, fallback_enabled, detail)


def _blocked_component_steps(fallback_enabled: bool, detail: str) -> tuple[SmokeStep, ...]:
    return (
        _blocked_step("component_model_load", fallback_enabled, detail),
        _blocked_step("component_text_encode", fallback_enabled, detail),
        _blocked_step("component_audio_encode", fallback_enabled, detail),
        _blocked_step("component_vae_encode_decode", fallback_enabled, detail),
        _blocked_step("component_minimal_generation", fallback_enabled, detail),
    )


def _result(
    environment: dict[str, str],
    steps: list[SmokeStep],
    blockers: list[str],
) -> SmokeMatrixResult:
    status = STATUS_PASS if steps and all(step.status == STATUS_PASS for step in steps) else STATUS_BLOCKED
    deduped_blockers = tuple(dict.fromkeys(blockers))
    return SmokeMatrixResult(status=status, environment=environment, steps=tuple(steps), blockers=deduped_blockers)
