from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "longcat_avatar_schema_test"


@dataclass(frozen=True)
class FakePort:
    direction: str
    port_type: str
    name: str | None = None
    display_name: str | None = None
    options: list[str] | None = None
    default: Any = None
    min: Any = None
    max: Any = None
    step: Any = None
    multiline: bool | None = None
    optional: bool = False


@dataclass(frozen=True)
class FakeSchema:
    node_id: str
    display_name: str
    category: str
    inputs: list[FakePort] = field(default_factory=list)
    outputs: list[FakePort] = field(default_factory=list)
    hidden: list[Any] = field(default_factory=list)
    is_output_node: bool = False


class FakeNodeOutput(tuple):
    def __new__(cls, *values: Any, ui: Any = None, **kwargs: Any):
        return super().__new__(cls, values)

    def __init__(self, *values: Any, ui: Any = None, **kwargs: Any):
        self.ui = ui


class _PortFactory:
    def __init__(self, port_type: str):
        self.port_type = port_type

    def Input(self, name: str, **kwargs: Any) -> FakePort:
        return FakePort(
            direction="input",
            port_type=self.port_type,
            name=name,
            options=kwargs.get("options"),
            default=kwargs.get("default"),
            min=kwargs.get("min"),
            max=kwargs.get("max"),
            step=kwargs.get("step"),
            multiline=kwargs.get("multiline"),
            optional=bool(kwargs.get("optional", False)),
        )

    def Output(self, display_name: str | None = None, **kwargs: Any) -> FakePort:
        return FakePort(
            direction="output",
            port_type=self.port_type,
            display_name=display_name,
            name=kwargs.get("name"),
        )


class _FakeIO:
    Schema = FakeSchema
    NodeOutput = FakeNodeOutput

    class ComfyNode:
        pass

    Combo = _PortFactory("COMBO")
    Model = _PortFactory("MODEL")
    Conditioning = _PortFactory("CONDITIONING")
    Image = _PortFactory("IMAGE")
    Int = _PortFactory("INT")
    Boolean = _PortFactory("BOOLEAN")
    Float = _PortFactory("FLOAT")
    String = _PortFactory("STRING")
    Clip = _PortFactory("CLIP")
    AudioEncoder = _PortFactory("AUDIO_ENCODER")
    Audio = _PortFactory("AUDIO")

    Hidden = types.SimpleNamespace(prompt="PROMPT", extra_pnginfo="EXTRA_PNGINFO")


class _FakeUI:
    class PreviewAudio:
        def __init__(self, audio: Any, cls: type | None = None, **kwargs: Any) -> None:
            self.audio = audio
            self.cls = cls

        def as_dict(self) -> dict[str, Any]:
            return {"audio": [{"filename": "preview.flac", "type": "temp"}]}


class _ModulePatcher:
    def __init__(self) -> None:
        self._old: dict[str, types.ModuleType | None] = {}

    def set(self, name: str, module: types.ModuleType) -> None:
        if name not in self._old:
            self._old[name] = sys.modules.get(name)
        sys.modules[name] = module

    def restore(self) -> None:
        for name, old_module in reversed(list(self._old.items())):
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_numpy_stub(patcher: _ModulePatcher) -> None:
    class _IInfo:
        max = 2**31 - 1

    numpy_stub = _module(
        "numpy",
        int32="int32",
        float32="float32",
        iinfo=lambda dtype: _IInfo(),
    )
    patcher.set("numpy", numpy_stub)


def _install_torch_stub(patcher: _ModulePatcher) -> None:
    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        max_memory_allocated=lambda: 0,
        ipc_collect=lambda: None,
    )
    mps = types.SimpleNamespace(is_available=lambda: False)
    torch_stub = _module(
        "torch",
        bfloat16="bfloat16",
        cuda=cuda,
        backends=types.SimpleNamespace(mps=mps),
        device=lambda name: name,
    )
    patcher.set("torch", torch_stub)


def _install_comfy_api_stub(patcher: _ModulePatcher) -> None:
    class ComfyExtension:
        pass

    latest = _module("comfy_api.latest", io=_FakeIO, ui=_FakeUI, ComfyExtension=ComfyExtension)
    patcher.set("comfy_api", _module("comfy_api"))
    patcher.set("comfy_api.latest", latest)


def _install_typing_extensions_stub(patcher: _ModulePatcher) -> None:
    def override(func: Any) -> Any:
        return func

    patcher.set("typing_extensions", _module("typing_extensions", override=override))


def _install_folder_paths_stub(patcher: _ModulePatcher, models_dir: str) -> None:
    fake_lists = {
        "diffusion_models": ["LongCat-Video-Avatar-1.5-int8.safetensors"],
        "gguf": ["unsupported-avatar.gguf"],
        "vae": ["LongCat-Video-Avatar-vae.safetensors"],
        "loras": ["longcat-avatar-dmd_lora.safetensors"],
        "audio_encoders": ["whisper-large-v3.safetensors"],
        "longcat": [
            "vocal.onnx",
            "LongCat-Video-Avatar-1.5/base_model/diffusion_pytorch_model.safetensors.index.json",
            "LongCat-Video-Avatar-1.5/base_model_int8/quantized_model.safetensors.index.json",
        ],
    }

    def get_filename_list(kind: str) -> list[str]:
        return list(fake_lists.get(kind, []))

    def get_full_path(kind: str, name: str) -> str:
        return str(Path(models_dir) / kind / name)

    folder_paths = _module(
        "folder_paths",
        models_dir=models_dir,
        add_model_folder_path=lambda kind, path: None,
        get_filename_list=get_filename_list,
        get_full_path=get_full_path,
        get_full_path_or_raise=get_full_path,
        get_output_directory=lambda: str(Path(models_dir) / "output"),
        get_temp_directory=lambda: str(Path(models_dir) / "temp"),
    )
    patcher.set("folder_paths", folder_paths)


def _install_runtime_stubs(patcher: _ModulePatcher) -> None:
    node_utils = _module(
        f"{PACKAGE_NAME}.node_utils",
        clear_comfyui_cache=lambda: None,
        tensor2image=lambda value: value,
        audio2path=lambda value: "audio.wav",
    )
    single_demo = _module(
        f"{PACKAGE_NAME}.LongCat_Video.run_demo_avatar_single_audio_to_video",
        load_longcat_video_model=lambda *args, **kwargs: None,
        generate=lambda *args, **kwargs: None,
        get_audio_vocal=lambda *args, **kwargs: ("audio.wav", None),
        get_audio_emb=lambda *args, **kwargs: {},
        load_audio_vocal=lambda *args, **kwargs: None,
    )
    multi_demo = _module(
        f"{PACKAGE_NAME}.LongCat_Video.run_demo_avatar_multi_audio_to_video",
        generate_multi=lambda *args, **kwargs: None,
    )
    patcher.set(f"{PACKAGE_NAME}.node_utils", node_utils)
    patcher.set(single_demo.__name__, single_demo)
    patcher.set(multi_demo.__name__, multi_demo)


@contextmanager
def loaded_longcat_node_module() -> Iterator[types.ModuleType]:
    patcher = _ModulePatcher()
    with tempfile.TemporaryDirectory() as models_dir:
        try:
            _install_numpy_stub(patcher)
            _install_torch_stub(patcher)
            _install_comfy_api_stub(patcher)
            _install_typing_extensions_stub(patcher)
            _install_folder_paths_stub(patcher, models_dir)

            package = _module(PACKAGE_NAME)
            package.__path__ = [str(REPO_ROOT)]  # type: ignore[attr-defined]
            patcher.set(PACKAGE_NAME, package)
            _install_runtime_stubs(patcher)

            module_name = f"{PACKAGE_NAME}.LongCat_Video_node"
            spec = importlib.util.spec_from_file_location(
                module_name,
                REPO_ROOT / "LongCat_Video_node.py",
            )
            if spec is None or spec.loader is None:
                raise ImportError("Could not build a module spec for LongCat_Video_node.py.")
            module = importlib.util.module_from_spec(spec)
            patcher.set(module_name, module)
            spec.loader.exec_module(module)
            yield module
        finally:
            patcher.restore()


@contextmanager
def loaded_longcat_extension_module() -> Iterator[types.ModuleType]:
    patcher = _ModulePatcher()
    with tempfile.TemporaryDirectory() as models_dir:
        try:
            _install_numpy_stub(patcher)
            _install_torch_stub(patcher)
            _install_comfy_api_stub(patcher)
            _install_typing_extensions_stub(patcher)
            _install_folder_paths_stub(patcher, models_dir)
            _install_runtime_stubs(patcher)

            spec = importlib.util.spec_from_file_location(
                PACKAGE_NAME,
                REPO_ROOT / "__init__.py",
                submodule_search_locations=[str(REPO_ROOT)],
            )
            if spec is None or spec.loader is None:
                raise ImportError("Could not build a module spec for __init__.py.")
            module = importlib.util.module_from_spec(spec)
            patcher.set(PACKAGE_NAME, module)
            spec.loader.exec_module(module)
            yield module
        finally:
            patcher.restore()
