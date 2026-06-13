from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised by repo tests without torch installed.
    # CRITICAL: keep torch optional at import time; CI contract tests run without PyTorch installed.
    torch = None


_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class BackendOperationResult:
    backend: str
    operation: str
    success: bool
    detail: str


@dataclass(frozen=True)
class BackendMemoryStats:
    backend: str
    allocated_bytes: int | None = None
    reserved_bytes: int | None = None
    max_allocated_bytes: int | None = None
    driver_allocated_bytes: int | None = None
    recommended_max_bytes: int | None = None
    detail: str = ""

    @property
    def available(self) -> bool:
        return any(
            value is not None
            for value in (
                self.allocated_bytes,
                self.reserved_bytes,
                self.max_allocated_bytes,
                self.driver_allocated_bytes,
                self.recommended_max_bytes,
            )
        )


@dataclass(frozen=True)
class BFloat16ProbeResult:
    backend: str
    supported: bool
    detail: str


@dataclass(frozen=True)
class BackendCapabilities:
    device: str
    backend: str
    available: bool
    built: bool | None
    supports_non_blocking: bool
    bfloat16: BFloat16ProbeResult
    mps_cpu_fallback_enabled: bool


def normalize_device_name(device: Any) -> str:
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None)
    if device_type:
        if device_index is None:
            return str(device_type).lower()
        return f"{str(device_type).lower()}:{device_index}"
    if device is None:
        return "cpu"
    return str(device).lower()


def normalize_backend_type(device: Any) -> str:
    normalized = normalize_device_name(device)
    return normalized.split(":", 1)[0]


def mps_cpu_fallback_enabled(environ: Mapping[str, str] | None = None) -> bool:
    value = (environ or os.environ).get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    return str(value).strip().lower() not in _FALSE_ENV_VALUES


def device_supports_non_blocking(device: Any) -> bool:
    return normalize_backend_type(device) == "cuda"


def _call_bool(obj: Any, name: str) -> bool | None:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return None
    try:
        return bool(fn())
    except Exception:
        return False


def _backend_available(backend: str, torch_module: Any) -> bool:
    if backend == "cpu":
        return True
    if backend == "cuda":
        return bool(_call_bool(getattr(torch_module, "cuda", None), "is_available"))
    if backend == "mps":
        mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        return bool(_call_bool(mps_backend, "is_available"))
    return False


def _backend_built(backend: str, torch_module: Any) -> bool | None:
    if backend != "mps":
        return None
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    return _call_bool(mps_backend, "is_built")


def describe_backend(
    device: Any,
    *,
    torch_module: Any = torch,
    environ: Mapping[str, str] | None = None,
) -> BackendCapabilities:
    backend = normalize_backend_type(device)
    return BackendCapabilities(
        device=normalize_device_name(device),
        backend=backend,
        available=_backend_available(backend, torch_module),
        built=_backend_built(backend, torch_module),
        supports_non_blocking=device_supports_non_blocking(device),
        bfloat16=probe_bfloat16_support(device, torch_module=torch_module),
        mps_cpu_fallback_enabled=mps_cpu_fallback_enabled(environ),
    )


def _operation_unavailable(backend: str, operation: str, detail: str) -> BackendOperationResult:
    return BackendOperationResult(backend=backend, operation=operation, success=False, detail=detail)


def _operation_success(backend: str, operation: str) -> BackendOperationResult:
    return BackendOperationResult(backend=backend, operation=operation, success=True, detail="ok")


def empty_cache(device: Any, *, torch_module: Any = torch) -> BackendOperationResult:
    backend = normalize_backend_type(device)
    operation = "empty_cache"
    if backend == "cpu":
        return _operation_unavailable(backend, operation, "cpu has no backend cache API")
    if backend == "cuda":
        cuda = getattr(torch_module, "cuda", None)
        if not bool(_call_bool(cuda, "is_available")):
            return _operation_unavailable(backend, operation, "cuda is unavailable")
        fn = getattr(cuda, "empty_cache", None)
    elif backend == "mps":
        mps = getattr(torch_module, "mps", None)
        fn = getattr(mps, "empty_cache", None)
    else:
        return _operation_unavailable(backend, operation, f"unsupported backend: {backend}")

    if not callable(fn):
        return _operation_unavailable(backend, operation, f"{backend} empty_cache API is unavailable")
    try:
        fn()
    except Exception as exc:
        return _operation_unavailable(backend, operation, f"{type(exc).__name__}: {exc}")
    return _operation_success(backend, operation)


def synchronize(device: Any, *, torch_module: Any = torch) -> BackendOperationResult:
    backend = normalize_backend_type(device)
    operation = "synchronize"
    if backend == "cpu":
        return _operation_unavailable(backend, operation, "cpu has no backend synchronize API")
    if backend == "cuda":
        cuda = getattr(torch_module, "cuda", None)
        if not bool(_call_bool(cuda, "is_available")):
            return _operation_unavailable(backend, operation, "cuda is unavailable")
        fn = getattr(cuda, "synchronize", None)
    elif backend == "mps":
        mps = getattr(torch_module, "mps", None)
        fn = getattr(mps, "synchronize", None)
    else:
        return _operation_unavailable(backend, operation, f"unsupported backend: {backend}")

    if not callable(fn):
        return _operation_unavailable(backend, operation, f"{backend} synchronize API is unavailable")
    try:
        if backend == "cuda":
            try:
                fn(device=device)
            except TypeError:
                fn()
        else:
            fn()
    except Exception as exc:
        return _operation_unavailable(backend, operation, f"{type(exc).__name__}: {exc}")
    return _operation_success(backend, operation)


def _read_int_api(obj: Any, name: str) -> tuple[int | None, str | None]:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return None, f"{name} unavailable"
    try:
        return int(fn()), None
    except Exception as exc:
        return None, f"{name} failed: {type(exc).__name__}: {exc}"


def read_memory_stats(device: Any, *, torch_module: Any = torch) -> BackendMemoryStats:
    backend = normalize_backend_type(device)
    details: list[str] = []
    if backend == "cuda":
        cuda = getattr(torch_module, "cuda", None)
        if not bool(_call_bool(cuda, "is_available")):
            return BackendMemoryStats(backend=backend, detail="cuda is unavailable")
        allocated, detail = _read_int_api(cuda, "memory_allocated")
        if detail:
            details.append(detail)
        reserved, detail = _read_int_api(cuda, "memory_reserved")
        if detail:
            details.append(detail)
        max_allocated, detail = _read_int_api(cuda, "max_memory_allocated")
        if detail:
            details.append(detail)
        return BackendMemoryStats(
            backend=backend,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            max_allocated_bytes=max_allocated,
            detail="; ".join(details),
        )
    if backend == "mps":
        mps = getattr(torch_module, "mps", None)
        allocated, detail = _read_int_api(mps, "current_allocated_memory")
        if detail:
            details.append(detail)
        driver, detail = _read_int_api(mps, "driver_allocated_memory")
        if detail:
            details.append(detail)
        recommended, detail = _read_int_api(mps, "recommended_max_memory")
        if detail:
            details.append(detail)
        return BackendMemoryStats(
            backend=backend,
            allocated_bytes=allocated,
            driver_allocated_bytes=driver,
            recommended_max_bytes=recommended,
            detail="; ".join(details),
        )
    return BackendMemoryStats(backend=backend, detail=f"{backend} memory counters are unavailable")


def _bytes_to_gb(value: int | None) -> str | None:
    if value is None:
        return None
    return f"{value / 1000 ** 3:.2f}"


def format_memory_fields(stats: BackendMemoryStats) -> list[str]:
    if stats.backend == "cuda":
        return [
            field
            for field in (
                f"cuda_alloc_gb={_bytes_to_gb(stats.allocated_bytes)}" if stats.allocated_bytes is not None else None,
                f"cuda_reserved_gb={_bytes_to_gb(stats.reserved_bytes)}" if stats.reserved_bytes is not None else None,
                f"cuda_max_alloc_gb={_bytes_to_gb(stats.max_allocated_bytes)}"
                if stats.max_allocated_bytes is not None
                else None,
            )
            if field is not None
        ]
    if stats.backend == "mps":
        return [
            field
            for field in (
                f"mps_alloc_gb={_bytes_to_gb(stats.allocated_bytes)}" if stats.allocated_bytes is not None else None,
                f"mps_driver_gb={_bytes_to_gb(stats.driver_allocated_bytes)}"
                if stats.driver_allocated_bytes is not None
                else None,
                f"mps_recommended_max_gb={_bytes_to_gb(stats.recommended_max_bytes)}"
                if stats.recommended_max_bytes is not None
                else None,
            )
            if field is not None
        ]
    return []


def move_to_device(value: Any, device: Any, *, dtype: Any = None) -> Any:
    kwargs = {"non_blocking": device_supports_non_blocking(device)}
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        return value.to(device, **kwargs)
    except TypeError:
        kwargs.pop("non_blocking", None)
        return value.to(device, **kwargs)


def probe_bfloat16_support(device: Any, *, torch_module: Any = torch) -> BFloat16ProbeResult:
    backend = normalize_backend_type(device)
    if not _backend_available(backend, torch_module):
        return BFloat16ProbeResult(backend=backend, supported=False, detail=f"{backend} is unavailable")
    dtype = getattr(torch_module, "bfloat16", None)
    if dtype is None:
        return BFloat16ProbeResult(backend=backend, supported=False, detail="torch.bfloat16 is unavailable")
    empty = getattr(torch_module, "empty", None)
    if not callable(empty):
        return BFloat16ProbeResult(backend=backend, supported=False, detail="torch.empty is unavailable")
    operations: list[tuple[str, Any]] = [("empty", lambda: empty((1,), device=device, dtype=dtype))]
    if backend == "mps":
        operations.extend(_mps_bfloat16_probe_operations(device, dtype, torch_module=torch_module))

    passed: list[str] = []
    for operation, probe_fn in operations:
        try:
            probe_fn()
        except Exception as exc:
            passed_detail = ", ".join(passed) if passed else "none"
            return BFloat16ProbeResult(
                backend=backend,
                supported=False,
                detail=f"{operation} failed after {passed_detail}: {type(exc).__name__}: {exc}",
            )
        passed.append(operation)
    return BFloat16ProbeResult(backend=backend, supported=True, detail=f"ok: {', '.join(passed)}")


def _mps_bfloat16_probe_operations(device: Any, dtype: Any, *, torch_module: Any) -> list[tuple[str, Any]]:
    ones = getattr(torch_module, "ones", None)
    arange = getattr(torch_module, "arange", None)
    nn_module = getattr(torch_module, "nn", None)
    conv3d_cls = getattr(nn_module, "Conv3d", None)
    if not callable(ones):
        return [("ones_api", lambda: (_ for _ in ()).throw(RuntimeError("torch.ones is unavailable")))]
    if not callable(arange):
        return [("arange_api", lambda: (_ for _ in ()).throw(RuntimeError("torch.arange is unavailable")))]
    if conv3d_cls is None:
        return [("conv3d_api", lambda: (_ for _ in ()).throw(RuntimeError("torch.nn.Conv3d is unavailable")))]

    def cpu_to_device() -> Any:
        return ones((2, 2), dtype=dtype).to(device=device)

    def add() -> Any:
        lhs = ones((2, 2), device=device, dtype=dtype)
        rhs = ones((2, 2), device=device, dtype=dtype)
        return lhs + rhs

    def matmul() -> Any:
        lhs = ones((4, 4), device=device, dtype=dtype)
        rhs = ones((4, 4), device=device, dtype=dtype)
        return lhs @ rhs

    def arange_bfloat16() -> Any:
        return arange(8, device=device, dtype=dtype)

    def conv3d() -> Any:
        conv = conv3d_cls(1, 1, kernel_size=1).to(device=device, dtype=dtype)
        sample = ones((1, 1, 1, 2, 2), device=device, dtype=dtype)
        return conv(sample)

    return [
        ("cpu_to_device", cpu_to_device),
        ("add", add),
        ("matmul", matmul),
        ("arange", arange_bfloat16),
        ("conv3d", conv3d),
    ]
