from __future__ import annotations

from dataclasses import dataclass

from .checkpoint_contract import (
    OFFICIAL_INT8_SHARDED,
    OFFICIAL_SHARDED,
    SINGLE_FILE_SAFETENSORS,
)


@dataclass(frozen=True)
class DitLoadPlan:
    loader_kind: str
    checkpoint_root: str | None
    subfolder: str | None
    single_file: str | None
    shard_paths: tuple[str, ...]
    use_int8: bool


def resolve_dit_load_plan(contract: object) -> DitLoadPlan:
    source_kind = getattr(contract, "source_kind", SINGLE_FILE_SAFETENSORS)
    use_int8 = bool(getattr(contract, "use_int8", False))

    if source_kind == SINGLE_FILE_SAFETENSORS:
        model_path = getattr(contract, "model_path", None)
        return DitLoadPlan(
            loader_kind="single_file_int8" if use_int8 else "single_file_safetensors",
            checkpoint_root=None,
            subfolder=None,
            single_file=model_path,
            shard_paths=(),
            use_int8=use_int8,
        )

    if source_kind == OFFICIAL_SHARDED:
        return DitLoadPlan(
            loader_kind="official_sharded",
            checkpoint_root=getattr(contract, "checkpoint_root", None),
            subfolder=getattr(contract, "checkpoint_subfolder", "base_model"),
            single_file=None,
            shard_paths=tuple(getattr(contract, "checkpoint_shard_paths", ()) or ()),
            use_int8=False,
        )

    if source_kind == OFFICIAL_INT8_SHARDED:
        return DitLoadPlan(
            loader_kind="official_sharded_int8",
            checkpoint_root=getattr(contract, "checkpoint_root", None),
            subfolder=getattr(contract, "checkpoint_subfolder", "base_model_int8"),
            single_file=None,
            shard_paths=tuple(getattr(contract, "checkpoint_shard_paths", ()) or ()),
            use_int8=True,
        )

    raise ValueError(f"Unsupported DiT load source: {source_kind}")
