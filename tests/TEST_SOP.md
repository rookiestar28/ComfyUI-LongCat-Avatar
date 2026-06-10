# Test SOP

Date: 2026-06-09
Status: Canonical repository test standard

This SOP defines the default validation standard for ComfyUI-LongCat-Avatar implementation work.

The repository integrates a large CUDA-oriented LongCat Avatar runtime into a ComfyUI custom node package. Full model inference is expensive and hardware-dependent, so the default acceptance gate emphasizes CPU-only contract tests, schema tests, workflow validation, static compilation, and public-safe evidence. GPU inference may be added as optional evidence, but it is not required unless an item-specific plan says so.

## 1. Source Of Truth

Follow this order:

1. Explicit user instruction in the current session.
2. `AGENTS.md`.
3. Item-specific `.planning/*_PLAN.md` acceptance criteria.
4. `ROADMAP.md`.
5. This SOP.
6. Existing test behavior.

If a higher-priority source overrides this SOP, document the override in the implementation record and command log.

## 2. Test Philosophy

Default tests must be:

- CPU-runnable by default.
- Deterministic and small.
- Free of real checkpoint downloads.
- Free of real GPU inference unless explicitly requested.
- Isolated from external reference repositories.
- Focused on contracts this repo controls: node schemas, path validation, checkpoint layout validation, audio/sampler/model payloads, and workflows.

Reference repositories under `reference/` are read-only research inputs. Tests must not import, execute, build, install, or invoke code from those repositories.

## 3. Environment

All repository tests must use the project-local Python environment for the OS/shell where the work is being executed.

Select and record the repo-local interpreter before running any tests:

```bash
# WSL/Linux bash
REPO_PYTHON=.venv-wsl/bin/python
$REPO_PYTHON --version
```

```powershell
# Windows PowerShell
$env:REPO_PYTHON = ".venv\Scripts\python.exe"
& $env:REPO_PYTHON --version
```

You may use bare `python` only after activating the repository venv in the current shell:

```bash
# WSL/Linux bash
source .venv-wsl/bin/activate
command -v python
python --version
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
Get-Command python
python --version
```

Command logs must record the interpreter path/version. Prefer explicit repo-local interpreter commands or a recorded `REPO_PYTHON` variable because they survive shell activation differences.

If the repo-local venv for the current OS is unavailable, stop and create or repair it before running tests unless the current user explicitly authorizes a temporary override. Any override must record the interpreter path/version and the reason it was not the repo venv.

Command examples below use POSIX shell `$REPO_PYTHON` syntax. In Windows PowerShell, run the equivalent command with `& $env:REPO_PYTHON`, for example `& $env:REPO_PYTHON -m unittest discover`.

Do not require real model weights for default tests. Use fake tensors, fixtures, small temporary files, and stubs.

## 4. Test Categories

### Contract Tests

Contract tests validate pure-Python behavior and data boundaries:

- model and checkpoint contracts;
- sharded checkpoint layout validation;
- download manifest decisions with mocked/no-network behavior;
- audio payload shape, duration, stride, and finite-value validation;
- sampler parameter normalization and output tensor contracts;
- bounding-box parsing;
- performance/memory mode normalization.

Typical files:

- `tests/test_model_contract.py`
- `tests/test_sharded_checkpoint_contract.py`
- `tests/test_model_download_contract.py`
- `tests/test_audio_contract.py`
- `tests/test_sampler_contract.py`
- `tests/test_bbox_contract.py`
- `tests/test_performance_contract.py`

### Node Schema Tests

Node schema tests validate that ComfyUI node definitions can be constructed using local stubs, without importing a real ComfyUI runtime or executing model inference.

Typical files:

- `tests/test_node_schema_contract.py`
- `tests/support/comfy_stubs.py`

### Workflow Tests

Workflow tests validate example workflow structure.

They must reject:

- private absolute paths;
- `.planning/`, `reference/`, `.sessions/`, or internal-only paths;
- arbitrary URL/token/raw JSON model inputs where unsupported;
- public claims for unsupported FP16/FP8/GGUF/scheduler behavior.

Typical files:

- `tests/test_workflow_contract.py`

### Static And Hygiene Gates

Static gates catch syntax and formatting hazards:

- `$REPO_PYTHON -m py_compile $(git ls-files '*.py')`
- `git diff --check`

Pre-commit gates are required when available:

- `pre-commit run detect-secrets --all-files`
- `pre-commit run --all-files --show-diff-on-failure`

If pre-commit is unavailable in the local environment, record the exact error and do not claim that pre-commit passed.

## 5. Targeted Test Selection

Use targeted tests before the full sweep. Item-specific plans may refine this list.

Model/checkpoint/download changes:

```bash
$REPO_PYTHON -m unittest tests.test_model_contract tests.test_sharded_checkpoint_contract tests.test_model_download_contract
```

Node UI/schema/workflow changes:

```bash
$REPO_PYTHON -m unittest tests.test_node_schema_contract tests.test_workflow_contract
```

Audio changes:

```bash
$REPO_PYTHON -m unittest tests.test_audio_contract tests.test_bbox_contract
```

Sampler changes:

```bash
$REPO_PYTHON -m unittest tests.test_sampler_contract tests.test_audio_contract
```

Performance/memory changes:

```bash
$REPO_PYTHON -m unittest tests.test_performance_contract tests.test_model_contract
```

## 6. Full Repository Gate

Run the full gate for every non-documentation implementation item before acceptance:

```bash
$REPO_PYTHON -m unittest discover
$REPO_PYTHON -m py_compile $(git ls-files '*.py')
git diff --check
pre-commit run detect-secrets --all-files
pre-commit run --all-files --show-diff-on-failure
git status --short --ignored
```

Documentation-only changes may skip test execution when they do not modify code, tests, scripts, infrastructure behavior, runtime config, workflow JSON, or generated artifacts. Public README/docs are intentionally not bound to automated tests; review them manually for accuracy and public-safety.

If workflow JSON files are touched, run `tests.test_workflow_contract`.

## 7. Prohibited Test Behavior

Default tests must not:

- download real model weights;
- require Hugging Face credentials;
- require CUDA/GPU availability;
- import or execute code from `reference/`;
- execute shell scripts from external repositories;
- write into `reference/`;
- write outside temporary directories or controlled ComfyUI output/model paths;
- log secrets, tokens, private URLs, private hostnames, cookies, or raw auth material;
- depend on local absolute paths.

## 8. Fixtures And Stubs

Use small fake objects and temporary directories for:

- tensor-like shape/finite checks;
- ComfyUI node schema construction;
- checkpoint index JSON;
- safetensors shard placeholders;
- download manifest decisions;
- output path validation.

Fixtures should be minimal and deterministic. Do not vendor large model files, generated binaries, or reference repository artifacts.

## 9. Evidence Requirements

Implementation records and command logs must include:

- date/timezone;
- workspace path;
- branch and commit SHA;
- OS/shell;
- Python path/version;
- exact commands;
- exit status;
- materially relevant redacted output;
- PASS/FAIL result;
- skipped or unavailable tools with exact reason.

For accepted non-documentation work, final verification evidence must map back to the roadmap/item acceptance criteria.

## 10. Acceptance Rule

An item is not accepted until:

- targeted tests relevant to the changed behavior pass;
- the full repository gate passes, or unavailable tools are recorded exactly according to the item plan;
- implementation record and command log evidence exist for non-documentation work;
- public workflow artifacts remain secret-free and do not expose internal paths;
- `git status --short --ignored` confirms internal-only paths remain ignored/untracked.
