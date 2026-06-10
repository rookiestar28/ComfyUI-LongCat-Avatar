from pathlib import Path
import importlib.util
import logging
import sys
import types
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTorch:
    def __init__(self):
        self.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=self._empty_cache,
            max_memory_allocated=lambda: 123_000_000,
        )
        self.device = lambda name: f"device:{name}"
        self.empty_cache_calls = 0

    def _empty_cache(self):
        self.empty_cache_calls += 1


class FakeModelManagement:
    def __init__(self):
        self.models = []
        self.soft_empty_cache_calls = 0
        self.runtime_device = "cuda:2"

    def get_torch_device(self):
        return self.runtime_device

    def loaded_models(self):
        return list(self.models)

    def soft_empty_cache(self):
        self.soft_empty_cache_calls += 1


class FakeModel:
    def __init__(self, should_raise=False):
        self.should_raise = should_raise
        self.unpatch_calls = []

    def unpatch_model(self, device_to):
        self.unpatch_calls.append(device_to)
        if self.should_raise:
            raise RuntimeError("boom")


class RuntimeDeviceCacheTests(unittest.TestCase):
    def load_node_utils(self):
        fake_torch = FakeTorch()
        fake_mm = FakeModelManagement()
        fake_comfy = types.ModuleType("comfy")
        fake_comfy_utils = types.ModuleType("comfy.utils")
        fake_comfy_utils.common_upscale = lambda *args, **kwargs: None
        fake_model_management = types.ModuleType("comfy.model_management")
        fake_model_management.get_torch_device = fake_mm.get_torch_device
        fake_model_management.loaded_models = fake_mm.loaded_models
        fake_model_management.soft_empty_cache = fake_mm.soft_empty_cache
        fake_pil = types.ModuleType("PIL")
        fake_pil_image = types.ModuleType("PIL.Image")
        fake_pil_image.fromarray = lambda *args, **kwargs: None
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.array = lambda value: value
        fake_numpy.float32 = "float32"
        fake_soundfile = types.ModuleType("soundfile")
        fake_soundfile.write = lambda *args, **kwargs: None
        fake_folder_paths = types.ModuleType("folder_paths")
        fake_folder_paths.get_temp_directory = lambda: "temp"

        module_name = "node_utils_runtime_test"
        installed = {
            "torch": fake_torch,
            "comfy": fake_comfy,
            "comfy.model_management": fake_model_management,
            "comfy.utils": fake_comfy_utils,
            "PIL": fake_pil,
            "PIL.Image": fake_pil_image,
            "numpy": fake_numpy,
            "soundfile": fake_soundfile,
            "folder_paths": fake_folder_paths,
        }
        old_modules = {name: sys.modules.get(name) for name in installed}
        try:
            sys.modules.update(installed)
            spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "node_utils.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module, fake_mm, fake_torch
        finally:
            sys.modules.pop(module_name, None)
            for name, old in old_modules.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old

    def test_runtime_device_uses_comfy_model_management(self):
        module, fake_mm, _ = self.load_node_utils()

        fake_mm.runtime_device = "cuda:7"

        self.assertEqual(module.get_runtime_device(), "cuda:7")

    def test_clear_cache_does_not_unpatch_loaded_models_by_default(self):
        module, fake_mm, fake_torch = self.load_node_utils()
        model = FakeModel()
        fake_mm.models = [model]

        module.clear_comfyui_cache()

        self.assertEqual(model.unpatch_calls, [])
        self.assertEqual(fake_mm.soft_empty_cache_calls, 1)
        self.assertEqual(fake_torch.empty_cache_calls, 1)

    def test_clear_cache_warning_for_targeted_unpatch_failure(self):
        module, fake_mm, _ = self.load_node_utils()
        fake_mm.models = [FakeModel(should_raise=True)]

        with self.assertLogs(level=logging.WARNING) as logs:
            module.clear_comfyui_cache(unload_loaded_models=True)

        self.assertIn("could not unpatch", "\n".join(logs.output))

    def test_node_source_has_no_module_level_device_hardcoding(self):
        source = (REPO_ROOT / "LongCat_Video_node.py").read_text(encoding="utf-8")

        self.assertNotIn('torch.device(\n    "cuda:0"', source)
        self.assertIn("runtime_device = get_runtime_device()", source)
        self.assertIn("build_runtime_plan(runtime_device", source)


if __name__ == "__main__":
    unittest.main()
