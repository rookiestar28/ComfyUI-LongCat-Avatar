# LongCat-Video Upstream Provenance

This directory contains a ComfyUI integration of selected LongCat-Video Avatar 1.5 runtime code.

## Upstream Source

- Upstream repository: <https://github.com/meituan-longcat/LongCat-Video>
- Reference commit: `6b3f4b8582a8bc3f20f795735f5383716c4ba794`
- Upstream project page: <https://meigen-ai.github.io/LongCat-Video-Avatar-1.5-Page/>
- License source: `LongCat_Video/LICENSE`

## Embedded Source Boundary

The following paths are treated as upstream-mirrored LongCat-Video runtime source with local compatibility patches:

- `LongCat_Video/longcat_video/`
- `LongCat_Video/run_demo_avatar_single_audio_to_video.py`
- `LongCat_Video/run_demo_avatar_multi_audio_to_video.py`
- `LongCat_Video/requirements.txt`
- `LongCat_Video/requirements_avatar.txt`

The following paths are local ComfyUI integration and contract layers:

- `LongCat_Video/audio_contract.py`
- `LongCat_Video/audio_crop.py`
- `LongCat_Video/attention_contract.py`
- `LongCat_Video/bbox_contract.py`
- `LongCat_Video/checkpoint_contract.py`
- `LongCat_Video/debug_profile.py`
- `LongCat_Video/layer_streaming.py`
- `LongCat_Video/model_contract.py`
- `LongCat_Video/model_loading_contract.py`
- `LongCat_Video/performance_contract.py`
- `LongCat_Video/sampler_contract.py`
- `LongCat_Video/scheduler_contract.py`
- `LongCat_Video/text_conditioning.py`
- `LongCat_Video/video_output.py`

## Local Patch Categories

Local patches are allowed only when they serve the ComfyUI integration contract:

- ComfyUI node input/output adaptation.
- Avatar 1.5 checkpoint and sharded-weight loading.
- Official text/audio conditioning parity.
- CUDA runtime, attention backend, and memory/offload controls.
- CPU-runnable contract tests and validation guards.
- Public documentation and workflow compatibility.

## Legacy Cleanup Policy

The first ComfyUI wrapper seed left compatibility shortcuts, stale comments, permissive fallbacks, and helper code that are not part of the supported Avatar 1.5 contract.

When touching a module in this directory, remove stale debris if it is not required by a supported workflow or covered compatibility contract. If legacy behavior must remain, document why in the implementation record and add a focused test for the retained behavior.

## Update Procedure

1. Compare upstream changes against the reference commit above.
2. Classify each change as upstream parity, ComfyUI adaptation, local bug fix, or unsupported/demo-only behavior.
3. Keep upstream-mirrored files and local contract files separate.
4. Update this file when the upstream reference commit changes.
5. Run the repository test gate before accepting the sync.

Do not copy generated caches, downloaded weights, private logs, local workflow exports, or reference-only artifacts into tracked source.
