import unittest
from pathlib import Path

from scripts.validate_comfy_registry_metadata import (
    EXPECTED_PACKAGE_NAME,
    EXPECTED_PUBLISHER_ID,
    EXPECTED_REPOSITORY,
    PINNED_PUBLISH_ACTION_RE,
    parse_simple_toml,
    toml_string,
    validate_repository,
)


ROOT = Path(__file__).resolve().parents[1]


class RegistryMetadataTests(unittest.TestCase):
    def test_pyproject_uses_finalized_comfy_registry_identity(self):
        pyproject = parse_simple_toml(ROOT / "pyproject.toml")

        self.assertEqual(toml_string(pyproject["project"]["name"]), EXPECTED_PACKAGE_NAME)
        self.assertEqual(toml_string(pyproject["project.urls"]["Repository"]), EXPECTED_REPOSITORY)
        self.assertEqual(toml_string(pyproject["tool.comfy"]["PublisherId"]), EXPECTED_PUBLISHER_ID)
        self.assertEqual(toml_string(pyproject["tool.comfy"]["DisplayName"]), "ComfyUI LongCat Avatar")
        self.assertIn("dependencies", toml_string(pyproject["project"]["dynamic"]))

    def test_pyproject_does_not_keep_upstream_publication_identity(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertNotIn("smthemex/ComfyUI_LongCat_Avatar", text)
        self.assertNotIn('PublisherId = "smthemex"', text)

    def test_registry_validator_accepts_current_repository(self):
        self.assertEqual(validate_repository(ROOT, require_finalized=True), [])

    def test_comfyignore_excludes_development_only_paths(self):
        text = (ROOT / ".comfyignore").read_text(encoding="utf-8")

        for required in (
            ".github/",
            ".planning/",
            ".sessions/",
            "reference/",
            "tests/",
            ".venv-wsl/",
            "AGENTS.md",
            "ROADMAP.md",
        ):
            self.assertIn(required, text)

    def test_publish_workflow_is_owner_gated_and_uses_registry_secret(self):
        workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")

        self.assertIn("REGISTRY_ACCESS_TOKEN", workflow)
        self.assertIn("github.repository_owner == 'rookiestar28'", workflow)
        self.assertIn("--require-finalized", workflow)
        self.assertRegex(workflow, PINNED_PUBLISH_ACTION_RE)
        self.assertNotIn("Comfy-Org/publish-node-action@main", workflow)


if __name__ == "__main__":
    unittest.main()
