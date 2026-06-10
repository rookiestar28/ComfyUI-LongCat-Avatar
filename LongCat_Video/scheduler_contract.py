from __future__ import annotations

from dataclasses import dataclass

from .model_contract import AVATAR_MAX_INFERENCE_STEPS, AVATAR_MIN_INFERENCE_STEPS

OFFICIAL_AVATAR_V15_SCHEDULER = "embedded_comfy_avatar_v15_scheduler"
OFFICIAL_AVATAR_V15_STEPS = 8
UNVERIFIED_REFERENCE_SCHEDULERS = ("longcat_distill_euler",)


@dataclass(frozen=True)
class AvatarSchedulerContract:
    scheduler_name: str
    steps: int
    status: str


def validate_avatar_scheduler_contract(
    *,
    scheduler_name: str | None = None,
    steps: int = OFFICIAL_AVATAR_V15_STEPS,
) -> AvatarSchedulerContract:
    scheduler_name = scheduler_name or OFFICIAL_AVATAR_V15_SCHEDULER
    steps = int(steps)
    if scheduler_name in UNVERIFIED_REFERENCE_SCHEDULERS:
        raise ValueError(
            f"Scheduler '{scheduler_name}' is a reference-wrapper scheduler and is not adopted "
            "as an official Avatar 1.5 parity scheduler in this repository."
        )
    if scheduler_name != OFFICIAL_AVATAR_V15_SCHEDULER:
        raise ValueError(
            f"Unsupported Avatar scheduler '{scheduler_name}'. "
            f"Expected '{OFFICIAL_AVATAR_V15_SCHEDULER}'."
        )
    if not AVATAR_MIN_INFERENCE_STEPS <= steps <= AVATAR_MAX_INFERENCE_STEPS:
        raise ValueError(
            "Avatar 1.5 sampler steps must be between "
            f"{AVATAR_MIN_INFERENCE_STEPS} and {AVATAR_MAX_INFERENCE_STEPS}."
        )
    return AvatarSchedulerContract(
        scheduler_name=scheduler_name,
        steps=steps,
        status="official_bounded",
    )


def scheduler_audit_summary() -> str:
    return (
        "Avatar 1.5 uses this repository's bounded official distill scheduler contract "
        f"with default {OFFICIAL_AVATAR_V15_STEPS} steps and allowed range "
        f"{AVATAR_MIN_INFERENCE_STEPS}-{AVATAR_MAX_INFERENCE_STEPS}. "
        "The reference-wrapper scheduler name longcat_distill_euler is documented as not adopted "
        "until official parity is proven with source comparison and tests."
    )
