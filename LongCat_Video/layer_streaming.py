"""Layer streaming wrapper for memory-efficient inference.
Keeps most transformer/decoder layers on CPU memory and moves active layers
to the target accelerator on demand. CUDA stream ownership is recorded only on
CUDA; MPS uses synchronous layer moves because it has no CUDA stream analogue.
General-purpose: works with any ``nn.Module`` whose forward iterates over a
``nn.ModuleList`` attribute (e.g. ``transformer_blocks``, ``layers``).
Each layer is evicted back to CPU immediately after its forward completes,
and active_count controls how many layers may remain resident.
Example
-------
>>> model = build_my_model(device=torch.device("cpu"))
>>> model = LayerStreamingWrapper(
...     model,
...     layers_attr="transformer_blocks",
...     target_device=torch.device("cuda:0"),
...     prefetch_count=2,
... )
>>> out = model(inputs)            # hooks handle layer streaming
>>> model.teardown()               # move everything back to CPU
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import functools
import gc
import itertools
import logging
from typing import Any, TypeVar

import torch
from torch import nn

from LongCat_Video.backend_capabilities import empty_cache, normalize_backend_type, synchronize

logger = logging.getLogger(__name__)
_M = TypeVar("_M", bound=torch.nn.Module)
T = TypeVar("T")


def cleanup_memory(device: torch.device | str = "cuda") -> None:
    gc.collect()
    empty_cache(device, torch_module=torch)
    synchronize(device, torch_module=torch)


def require_streaming_device(target_device: torch.device) -> None:
    if normalize_backend_type(target_device) not in {"cuda", "mps"}:
        raise RuntimeError(
            "Layer streaming requires a CUDA or MPS device."
        )


def _record_tensor_stream(tensor: torch.Tensor, target_device: torch.device) -> None:
    if normalize_backend_type(target_device) == "cuda":
        tensor.record_stream(torch.cuda.current_stream(target_device))

# LayerStreamingWrapper from https://github.com/Lightricks/LTX-2

class SimpleLayerStreamingWrapper_Dual(nn.Module):
    """简化版层流式处理包装器，支持多模块卸载"""

    def __init__(
        self,
        model: nn.Module,
        layers_attrs: list[str],  # 修改为列表，支持多个模块路径
        target_device: torch.device,
        active_count: int = 1,
    ) -> None:
        super().__init__()
        self._model = model
        self._layers_attrs = layers_attrs
        self._target_device = target_device
        self._active_count = active_count

        # 解析并存储所有需要卸载的模块
        self._layer_groups: list[nn.ModuleList] = []
        self._stores: list[_SimpleLayerStore] = []
        self._hook_handles: list[Any] = []

        for attr in self._layers_attrs:
            layers = _resolve_attr(model, attr)
            self._layer_groups.append(layers)
            self._stores.append(_SimpleLayerStore(layers, self._target_device))

        # 将非层参数移到GPU
        self._move_non_layer_params_to_gpu()

        # 为所有模块组注册钩子
        self._register_simple_hooks()

    def _move_non_layer_params_to_gpu(self) -> None:
        """移动非层参数到GPU，排除所有需要流式卸载的模块参数"""
        layer_tensor_ids = set()
        # 收集所有卸载模块的参数 ID
        for layers in self._layer_groups:
            for layer in layers:
                for t in itertools.chain(layer.parameters(), layer.buffers()):
                    layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """代理属性访问到原始模型"""
        try:
            # 首先尝试从包装器自身获取属性
            return super().__getattr__(name)
        except AttributeError:
            # 如果失败，则从原始模型获取
            return getattr(self._model, name)

    def _register_simple_hooks(self) -> None:
        """为所有模块组注册简单的加载/释放钩子"""
        # 遍历每一个模块组及其对应的 Store
        for layers, store in zip(self._layer_groups, self._stores):
            idx_map = {id(layer): idx for idx, layer in enumerate(layers)}

            def _pre_hook(module: nn.Module, input, *, idx: int, s: _SimpleLayerStore):
                # 加载当前层到GPU
                s.load_layer_to_gpu(idx, module)
                # 记录 CUDA stream，防止内存被提前回收；MPS has no CUDA stream equivalent.
                for param in itertools.chain(module.parameters(), module.buffers()):
                    _record_tensor_stream(param.data, self._target_device)

            def _post_hook(module: nn.Module, input, output, *, idx: int, s: _SimpleLayerStore):
                # 处理完后立即将层移回CPU
                s.unload_layer_from_gpu(idx, module)

            for layer in layers:
                idx = idx_map[id(layer)]
                # 使用 functools.partial 将对应的 store 实例传入钩子
                pre_hook = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx, s=store))
                post_hook = layer.register_forward_hook(functools.partial(_post_hook, idx=idx, s=store))
                self._hook_handles.extend((pre_hook, post_hook))

    def teardown(self) -> None:
        # CRITICAL: remove streaming hooks; otherwise every segment stacks hooks and slows denoising.
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._model.to("cpu")

@contextmanager
def _streaming_model(
    model: _M,
    layers_attr,  # 允许接收 str 或 list[str]
    target_device: torch.device,
    prefetch_count: int,
) -> Iterator[_M]:
    """Wrap *model* with :class:`LayerStreamingWrapper`, yield it, then tear down."""
    require_streaming_device(target_device)
    # 根据传入的 layers_attr 类型自动路由到对应的 Wrapper
    if isinstance(layers_attr, list):
        wrapped = SimpleLayerStreamingWrapper_Dual(
            model,
            layers_attrs=layers_attr,
            target_device=target_device,
            active_count=prefetch_count,
        )
    else:
        wrapped = SimpleLayerStreamingWrapper(
            model,
            layers_attr=layers_attr,
            target_device=target_device,
            active_count=prefetch_count,
        )

    try:
        yield wrapped  # type: ignore[misc]
    finally:
        wrapped.teardown()
        cleanup_memory(target_device)
        try:
            if normalize_backend_type(target_device) == "cuda" and hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
        except Exception:
            print("Host empty cache cleanup failed; ignoring.", exc_info=True)



@contextmanager
def _streaming_model_(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
    prefetch_count: int,
) -> Iterator[_M]:
    """Wrap *model* with :class:`LayerStreamingWrapper`, yield it, then tear down."""
    require_streaming_device(target_device)
    wrapped = SimpleLayerStreamingWrapper(
        model,
        layers_attr=layers_attr,
        target_device=target_device,
        active_count=prefetch_count,
    )
    try:
        yield wrapped  # type: ignore[misc]
    finally:
        wrapped.teardown()
        cleanup_memory(target_device)
        # Flush the host (pinned) memory cache so that freed pinned pages are
        # returned to the OS.  Without this, sequential streaming models
        # (e.g. text encoder then transformer) exhaust host memory because the
        # CachingHostAllocator keeps freed blocks cached indefinitely.
        try:
            if normalize_backend_type(target_device) == "cuda" and hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
        except Exception:
            print("Host empty cache cleanup failed; ignoring.", exc_info=True)


def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj

# edit from LayerStreamingWrapper from https://github.com/Lightricks/LTX-2

class _SimpleLayerStore:
    """简化版层存储，支持按需加载和立即释放"""

    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)

        # 保留CPU端的原始参数引用
        self._cpu_params: list[dict[str, torch.Tensor]] = []
        for layer in layers:
            cpu_copy = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                cpu_copy[name] = tensor.data.cpu()  # 保留在CPU上
            self._cpu_params.append(cpu_copy)

    def load_layer_to_gpu(self, idx: int, layer: nn.Module) -> None:
        """将指定层加载到GPU"""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name].to(self.target_device)

    def unload_layer_from_gpu(self, idx: int, layer: nn.Module) -> None:
        """将指定层从GPU卸载回CPU"""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name]  # 恢复为CPU副本


class SimpleLayerStreamingWrapper(nn.Module):
    """简化版层流式处理包装器"""

    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        active_count: int = 1,  # 同时激活的层数量
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        self._active_count = active_count
        self._store = _SimpleLayerStore(self._layers, self._target_device)
        self._hook_handles: list[Any] = []

        # 将非层参数移到GPU
        self._move_non_layer_params_to_gpu()

        # 注册钩子
        self._register_simple_hooks()

    def _move_non_layer_params_to_gpu(self) -> None:
        """移动非层参数到GPU"""
        layer_tensor_ids = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)

    def _register_simple_hooks(self) -> None:
        """注册简单的加载/释放钩子"""
        idx_map = {id(layer): idx for idx, layer in enumerate(self._layers)}

        def _pre_hook(module: nn.Module, input, *, idx: int):
            # 加载当前层到GPU
            self._store.load_layer_to_gpu(idx, module)
            # 记录 CUDA stream，防止内存被提前回收；MPS has no CUDA stream equivalent.
            for param in itertools.chain(module.parameters(), module.buffers()):
                _record_tensor_stream(param.data, self._target_device)

        def _post_hook(module: nn.Module, input, output, *, idx: int):
            # 处理完后立即将层移回CPU
            self._store.unload_layer_from_gpu(idx, module)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            pre_hook = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            post_hook = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
            self._hook_handles.extend((pre_hook, post_hook))

    def teardown(self) -> None:
        # CRITICAL: remove streaming hooks; otherwise every segment stacks hooks and slows denoising.
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._model.to("cpu")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """代理属性访问到原始模型"""
        try:
            # 首先尝试从包装器自身获取属性
            return super().__getattr__(name)
        except AttributeError:
            # 如果失败，则从原始模型获取
            return getattr(self._model, name)
