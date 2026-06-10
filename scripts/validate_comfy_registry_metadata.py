#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


EXPECTED_PACKAGE_NAME = "comfyui-longcat-avatar"
EXPECTED_PUBLISHER_ID = "rookiestar"
EXPECTED_REPOSITORY = "https://github.com/rookiestar28/ComfyUI-LongCat-Avatar"
EXPECTED_OWNER = "rookiestar28"
PINNED_PUBLISH_ACTION_RE = re.compile(r"Comfy-Org/publish-node-action@[0-9a-f]{40}")


def _strip_inline_comment(line: str) -> str:
    in_quote = False
    quote = ""
    escaped = False
    result = []
    for char in line:
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\" and in_quote:
            result.append(char)
            escaped = True
            continue
        if char in ("'", '"'):
            if in_quote and char == quote:
                in_quote = False
                quote = ""
            elif not in_quote:
                in_quote = True
                quote = char
        if char == "#" and not in_quote:
            break
        result.append(char)
    return "".join(result).strip()


def parse_simple_toml(path: Path) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {"": {}}
    current = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_inline_comment(raw_line)
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]")
            sections.setdefault(current, {})
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        sections[current][key.strip()] = value.strip()
    return sections


def toml_string(raw: str | None) -> str:
    if raw is None:
        return ""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    return raw


def toml_file_reference(raw: str | None) -> str:
    if raw is None:
        return ""
    match = re.search(r'file\s*=\s*"([^"]+)"', raw)
    return match.group(1) if match else ""


def validate_pyproject(path: Path, *, require_finalized: bool) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    data = parse_simple_toml(path)
    project = data.get("project", {})
    urls = data.get("project.urls", {})
    comfy = data.get("tool.comfy", {})
    dynamic = data.get("tool.setuptools.dynamic", {})

    name = toml_string(project.get("name"))
    version = toml_string(project.get("version"))
    repository = toml_string(urls.get("Repository"))
    publisher_id = toml_string(comfy.get("PublisherId"))
    display_name = toml_string(comfy.get("DisplayName"))
    icon = toml_string(comfy.get("Icon"))
    license_file = toml_file_reference(project.get("license"))

    if name != EXPECTED_PACKAGE_NAME:
        errors.append(f"[project].name must be {EXPECTED_PACKAGE_NAME!r}.")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{0,98}", name):
        errors.append("[project].name must be a valid Comfy Registry node id.")
    if re.search(r"[._-]{2,}", name):
        errors.append("[project].name must not contain consecutive special characters.")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        errors.append("[project].version must use X.Y.Z semantic versioning.")
    if "readme" not in project:
        errors.append("[project].readme is required for registry metadata.")
    if "requires-python" not in project:
        errors.append("[project].requires-python is required for registry metadata.")
    if not license_file or not (path.parent / license_file).is_file():
        errors.append("[project].license must point to an existing LICENSE file.")
    if dynamic.get("dependencies") != '{file = ["requirements.txt"]}':
        errors.append('[tool.setuptools.dynamic].dependencies must read from "requirements.txt".')
    if publisher_id != EXPECTED_PUBLISHER_ID:
        errors.append(f"[tool.comfy].PublisherId must be {EXPECTED_PUBLISHER_ID!r}.")
    if not display_name:
        errors.append("[tool.comfy].DisplayName must be set.")
    if repository != EXPECTED_REPOSITORY:
        errors.append(f"[project.urls].Repository must be {EXPECTED_REPOSITORY!r}.")
    if icon and not re.match(r"https?://", icon) and not (path.parent / icon).is_file():
        errors.append("[tool.comfy].Icon must be a URL or an existing repository path.")
    if "smthemex/ComfyUI_LongCat_Avatar" in text or 'PublisherId = "smthemex"' in text:
        errors.append("Registry metadata must not retain stale upstream publication identity.")
    if require_finalized and "TODO" in text:
        errors.append("Finalized registry metadata must not contain TODO placeholders.")

    return errors


def validate_comfyignore(path: Path) -> list[str]:
    required_patterns = {
        ".github/",
        ".planning/",
        ".sessions/",
        "reference/",
        "tests/",
        ".venv-wsl/",
        "AGENTS.md",
        "ROADMAP.md",
    }
    if not path.is_file():
        return [".comfyignore is required for registry packaging."]
    patterns = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_patterns - patterns)
    return [f".comfyignore is missing required pattern: {pattern}" for pattern in missing]


def validate_workflow(path: Path) -> list[str]:
    if not path.is_file():
        return [".github/workflows/publish.yml is required."]
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if "REGISTRY_ACCESS_TOKEN" not in text:
        errors.append("Publish workflow must use the REGISTRY_ACCESS_TOKEN secret.")
    if f"github.repository_owner == '{EXPECTED_OWNER}'" not in text:
        errors.append("Publish workflow must gate execution to the expected repository owner.")
    if "--require-finalized" not in text:
        errors.append("Publish workflow must run finalized registry metadata validation.")
    if not PINNED_PUBLISH_ACTION_RE.search(text):
        errors.append("Publish workflow must pin Comfy-Org/publish-node-action to a commit SHA.")
    if "Comfy-Org/publish-node-action@main" in text:
        errors.append("Publish workflow must not use a floating publish action ref.")
    return errors


def validate_repository(root: Path, *, require_finalized: bool) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_pyproject(root / "pyproject.toml", require_finalized=require_finalized))
    errors.extend(validate_comfyignore(root / ".comfyignore"))
    errors.extend(validate_workflow(root / ".github" / "workflows" / "publish.yml"))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Comfy Registry publishing metadata.")
    parser.add_argument("--root", default=".", help="Repository root to validate.")
    parser.add_argument(
        "--require-finalized",
        action="store_true",
        help="Fail if registry metadata still contains placeholders.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    errors = validate_repository(root, require_finalized=args.require_finalized)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Comfy Registry metadata validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
