from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
