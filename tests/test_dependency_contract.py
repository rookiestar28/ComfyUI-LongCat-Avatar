from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).resolve().parents[1]


def requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "http:", "https:", "git+")):
            continue
        for marker in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            if marker in line:
                line = line.split(marker, 1)[0]
                break
        names.add(line.strip().lower().replace("_", "-"))
    return names


class DependencyContractTests(unittest.TestCase):
    def test_default_requirements_do_not_override_comfyui_cuda_stack(self):
        forbidden = {
            "torch",
            "torchvision",
            "torchaudio",
            "flash-attn",
            "streamlit",
            "openai",
            "pyarrow",
            "tritonserverclient",
        }

        self.assertFalse(requirement_names(ROOT / "requirements.txt") & forbidden)

    def test_default_requirements_include_exposed_node_runtime_dependencies(self):
        default = requirement_names(ROOT / "requirements.txt")
        self.assertIn("audio-separator", default)
        self.assertIn("onnx", default)
        self.assertIn("onnxruntime", default)

        acceleration_text = (ROOT / "requirements-acceleration.txt").read_text(encoding="utf-8")
        self.assertIn("flash-attn", acceleration_text)
        self.assertIn("xformers", acceleration_text)
        self.assertIn("SageAttention", acceleration_text)

    def test_audio_separator_is_not_imported_at_module_import_time(self):
        for path in (
            ROOT / "LongCat_Video" / "run_demo_avatar_single_audio_to_video.py",
            ROOT / "LongCat_Video" / "run_demo_avatar_multi_audio_to_video.py",
        ):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "audio_separator.separator")
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith("audio_separator") for alias in node.names))

        single_source = (ROOT / "LongCat_Video" / "run_demo_avatar_single_audio_to_video.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("requires vocal separation dependencies", single_source)
        self.assertIn("requirements.txt", single_source)
        self.assertNotIn("requirements-vocal.txt", single_source)

    def test_block_sparse_triton_import_is_optional_for_mps_node_startup(self):
        path = ROOT / "LongCat_Video" / "longcat_video" / "block_sparse_attention" / "bsa_interface.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name == "triton" for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "triton")
                self.assertNotEqual(node.module, "triton.language")

        self.assertIn("macOS/MPS must be able to import ComfyUI nodes without Triton", source)
        self.assertIn("Use the SDPA attention backend on macOS MPS", source)


if __name__ == "__main__":
    unittest.main()
