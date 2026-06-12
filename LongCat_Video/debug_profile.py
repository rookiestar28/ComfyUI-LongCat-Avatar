from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator

from LongCat_Video.backend_capabilities import format_memory_fields, read_memory_stats, synchronize


class LongCatDebugProfiler:
    def __init__(self, enabled: bool = False, *, label: str = "sampler", device: Any = None) -> None:
        self.enabled = bool(enabled)
        self.label = label
        self.device = device
        self._active_phases: dict[str, tuple[float, dict[str, Any]]] = {}

    @contextmanager
    def phase(self, name: str, **fields: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        self._sync_backend()
        start = perf_counter()
        self.log(f"{name}.start", **fields)
        try:
            yield
        finally:
            self._sync_backend()
            elapsed = perf_counter() - start
            self.log(f"{name}.end", elapsed_s=f"{elapsed:.3f}", **fields)

    def mark(self, name: str, **fields: Any) -> None:
        if self.enabled:
            self.log(name, **fields)

    def start_phase(self, name: str, **fields: Any) -> None:
        if not self.enabled:
            return
        self._sync_backend()
        self._active_phases[name] = (perf_counter(), fields)
        self.log(f"{name}.start", **fields)

    def end_phase(self, name: str, **fields: Any) -> None:
        if not self.enabled:
            return
        self._sync_backend()
        started = self._active_phases.pop(name, None)
        if started is None:
            self.log(f"{name}.end", elapsed_s="unknown", **fields)
            return
        start, start_fields = started
        merged_fields = dict(start_fields)
        merged_fields.update(fields)
        elapsed = perf_counter() - start
        self.log(f"{name}.end", elapsed_s=f"{elapsed:.3f}", **merged_fields)

    def child(self, label: str) -> "LongCatDebugProfiler":
        return LongCatDebugProfiler(self.enabled, label=f"{self.label}.{label}", device=self.device)

    def log(self, event: str, **fields: Any) -> None:
        parts = [f"[DEBUG] LongCat profile {self.label}.{event}"]
        parts.extend(f"{key}={value}" for key, value in fields.items() if value is not None)
        parts.extend(self._backend_memory_fields())
        print(" | ".join(parts))

    def _sync_backend(self) -> None:
        synchronize(self.device)

    def _backend_memory_fields(self) -> list[str]:
        return format_memory_fields(read_memory_stats(self.device))


DISABLED_DEBUG_PROFILER = LongCatDebugProfiler(False)


def ensure_debug_profiler(debug_profile: LongCatDebugProfiler | None) -> LongCatDebugProfiler:
    return debug_profile if debug_profile is not None else DISABLED_DEBUG_PROFILER
