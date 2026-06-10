import ast
import json
from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_RUNTIME_FILES = (
    ROOT / "LongCat_Video" / "checkpoint_contract.py",
    ROOT / "LongCat_Video" / "run_demo_avatar_single_audio_to_video.py",
    ROOT / "LongCat_Video" / "run_demo_avatar_multi_audio_to_video.py",
    ROOT / "LongCat_Video" / "longcat_video" / "pipeline_longcat_video.py",
    ROOT / "LongCat_Video" / "longcat_video" / "pipeline_longcat_video_avatar.py",
)
PUBLIC_GGUF_SURFACE_FILES = (
    ROOT / "LongCat_Video_node.py",
    ROOT / "node_utils.py",
    ROOT / "__init__.py",
)
PUBLIC_WORKFLOW_FILES = tuple((ROOT / "example_workflows").glob("*.json"))
PUBLIC_JS_EXTENSION_FILES = tuple((ROOT / "js").glob("*.js"))
FORBIDDEN_GGUF_PUBLIC_TOKENS = (
    'io.Combo.Input("gguf"',
    'add_model_folder_path("gguf"',
    "gguf_experimental_unsupported",
    "GGUFQuantizationConfig",
    "GGUFQuantizer",
    "GGUFReader",
    "load_gguf_checkpoint",
    "set_gguf2meta_model",
    "apply_loras_gguf",
)


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def _literal_true(node):
    return isinstance(node, ast.Constant) and node.value is True


def _shell_invocation_offenders(path):
    offenders = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        call_name = _call_name(node.func)
        has_shell_true = any(keyword.arg == "shell" and _literal_true(keyword.value) for keyword in node.keywords)
        if call_name == "os.system" or has_shell_true:
            offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{call_name or 'shell=True'}")
    return offenders


class QualityGateContractTests(unittest.TestCase):
    def test_pre_commit_has_pinned_scoped_ruff_hook(self):
        text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

        self.assertIn("https://github.com/astral-sh/ruff-pre-commit", text)
        self.assertRegex(text, r"rev:\s+v\d+\.\d+\.\d+")
        self.assertIn("id: ruff", text)
        self.assertIn("args: [--config=pyproject.toml]", text)

    def test_ruff_hook_scope_is_repo_owned_and_excludes_embedded_upstream(self):
        text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        match = re.search(r"files:\s+(.+)", text)
        self.assertIsNotNone(match)
        files_regex = match.group(1)

        self.assertRegex("LongCat_Video_node.py", files_regex)
        self.assertRegex("LongCat_Video/sampler_contract.py", files_regex)
        self.assertRegex("tests/test_sampler_contract.py", files_regex)
        self.assertRegex("scripts/validate_comfy_registry_metadata.py", files_regex)

        self.assertNotRegex("LongCat_Video/longcat_video/pipeline_longcat_video_avatar.py", files_regex)
        self.assertNotRegex("reference/ComfyUI_LongCat_Avatar/some_file.py", files_regex)
        self.assertNotRegex(".planning/internal.py", files_regex)
        self.assertNotRegex(".sessions/session.py", files_regex)

    def test_ruff_config_uses_guardrail_rules_without_formatter_churn(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        ruff = pyproject["tool"]["ruff"]
        lint = ruff["lint"]

        self.assertEqual(ruff["target-version"], "py39")
        self.assertIn("LongCat_Video/longcat_video", ruff["extend-exclude"])
        self.assertIn("reference", ruff["extend-exclude"])
        self.assertIn("E722", lint["select"])
        self.assertNotIn("format", ruff)

    def test_supported_runtime_boundaries_do_not_use_runtime_asserts(self):
        offenders = []
        for path in SUPPORTED_RUNTIME_FILES:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("assert ") or stripped == "assert":
                    offenders.append(f"{path.relative_to(ROOT)}:{line_number}:{stripped}")

        self.assertEqual(offenders, [])

    def test_supported_runtime_boundaries_do_not_invoke_shell_commands(self):
        offenders = []
        for path in SUPPORTED_RUNTIME_FILES:
            offenders.extend(_shell_invocation_offenders(path))

        self.assertEqual(offenders, [])

    def test_public_surface_does_not_reintroduce_unsupported_gguf_loading(self):
        text_files = PUBLIC_GGUF_SURFACE_FILES + PUBLIC_JS_EXTENSION_FILES
        offenders = []
        for path in text_files:
            source = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_GGUF_PUBLIC_TOKENS:
                if token in source:
                    offenders.append(f"{path.relative_to(ROOT)}:{token}")

        for path in PUBLIC_WORKFLOW_FILES:
            workflow = json.loads(path.read_text(encoding="utf-8"))
            workflow_text = json.dumps(workflow, sort_keys=True)
            for token in ("gguf", ".gguf", "unsupported-avatar.gguf"):
                if token in workflow_text:
                    offenders.append(f"{path.relative_to(ROOT)}:{token}")

        self.assertEqual(offenders, [])

    def test_debt_gate_scope_excludes_internal_reference_and_upstream_bulk(self):
        gated_paths = SUPPORTED_RUNTIME_FILES + PUBLIC_GGUF_SURFACE_FILES + PUBLIC_WORKFLOW_FILES + PUBLIC_JS_EXTENSION_FILES
        offenders = []
        for path in gated_paths:
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith((".planning/", "reference/", ".sessions/")):
                offenders.append(relative)
            if relative.startswith("LongCat_Video/longcat_video/modules/"):
                offenders.append(relative)

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
