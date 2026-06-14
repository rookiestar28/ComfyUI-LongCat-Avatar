# ComfyUI LongCat Avatar - macOS MLX Branch

This branch adapts LongCat-Video-Avatar 1.5 for ComfyUI on Apple Silicon through
an external Apple MLX runner. It follows the community
[longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx) port for
the Apple Silicon inference path while keeping the ComfyUI node boundary,
request/response validation, job-local logs, and content-free smoke evidence in
this repository.

The previous PyTorch MPS experiment is not the active support path for this
branch. It remains diagnostic work only; native PyTorch MPS generation did not
produce an accepted first artifact. The macOS path here is the MLX external
runner path.

<p align="center">
  <a href="https://github.com/meituan-longcat/LongCat-Video">Official LongCat-Video Repo</a> |
  <a href="https://meigen-ai.github.io/LongCat-Video-Avatar-1.5-Page/">Official Avatar 1.5 Project Page</a> |
  <a href="https://github.com/xocialize/longcat-avatar-mlx">longcat-avatar-mlx Reference Port</a>
</p>

<div align="center">
  <video src="https://github.com/user-attachments/assets/dbad0a81-c961-4cef-9a18-09377de2397a" width="900"></video>
</div>

## Status

| Capability | Status |
| --- | --- |
| Apple Silicon MLX external runner node | Implemented repo-side |
| MLX runner request/response schema | Implemented |
| MLX environment and weight preflight | Implemented |
| MLX job-local realtime runner log | Implemented |
| macOS unified-memory probe | Implemented |
| q4 support-gate smoke harness | Implemented repo-side; still requires a real 32 GB+ Apple Silicon artifact run before public support wording changes |
| PyTorch MPS generation | Not the active path; blocked by prior native-MPS denoising stall evidence |
| Avatar 1.5 single-audio generation | Target MLX path |
| Avatar 1.0 | Not supported by this ComfyUI contract |

## Recommended Hardware

The MLX reference port reports Apple Silicon M-series testing on an M5 Max with
128 GB unified memory. For this ComfyUI branch, use the q4 model first unless you
have a larger Mac.

| Variant | Disk | Suggested Mac | Notes |
| --- | ---: | --- | --- |
| `q4-dmd-merged` | ~24 GB | 32-48 GB unified memory recommended | 4-bit DiT, DMD pre-merged, current first smoke-gate target |
| `q8-dmd-merged` | ~31 GB | 32 GB+ unified memory | 8-bit DiT, middle-ground RAM/quality |
| `bf16-dmd-merged` | ~43 GB | 64 GB+ unified memory recommended | DMD pre-merged bf16 |
| `bf16` | ~46 GB | 64 GB+ unified memory recommended | Base bf16 plus separate DMD LoRA, mainly for experiments |

Reference performance from `longcat-avatar-mlx`, measured at 256 x 432 x 29
frames with 8-step DMD sampling on Apple M5 Max 128 GB:

| Variant | Wall clock |
| --- | ---: |
| `bf16-dmd-merged` | ~105 s |
| `q4-dmd-merged` | ~102 s |
| `q8-dmd-merged` | ~151 s |

The q4 model card lists a lower minimum memory figure, but this ComfyUI branch
keeps the first artifact smoke gate conservative at 32 GB+ unified memory.

## Model Downloads

The MLX weights are published by `mlx-community`:

- Collection: <https://huggingface.co/collections/mlx-community/longcat-video-avatar-15-mlx-6a185d1af4a43074d882e375>
- q4 DMD merged: <https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged>
- q8 DMD merged: <https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q8-dmd-merged>
- bf16 DMD merged: <https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged>
- bf16 base: <https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16>

Recommended first download for this branch:

```bash
hf download mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged \
  --local-dir ComfyUI/models/longcat/LongCat-Video-Avatar-1.5-mlx/LongCat-Video-Avatar-1.5-q4-dmd-merged
```

The runner validates the expected MLX variant layout. If you use a parent
weights folder, keep each variant under its own directory and point the MLX node
at the parent folder.

Example layout:

```text
ComfyUI/models/longcat/LongCat-Video-Avatar-1.5-mlx/
  LongCat-Video-Avatar-1.5-q4-dmd-merged/
    vae/
    text_encoder/
    audio_encoder/
    dit/
    scheduler/
    tokenizer/
```

## Installation

Install this custom node inside ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rookiestar28/ComfyUI-LongCat-Avatar
cd ComfyUI-LongCat-Avatar
pip install -r requirements.txt
```

Use the Python environment that launches ComfyUI for the custom node itself.
Create a separate MLX runner environment for Apple Silicon inference:

```bash
python3.12 -m venv .venv-mlx
.venv-mlx/bin/pip install -U pip
.venv-mlx/bin/pip install \
  mlx safetensors huggingface-hub numpy transformers pillow \
  imageio imageio-ffmpeg librosa mlx-arsenal
```

Install the external MLX runtime package used by the runner. During development
this branch tracks the `longcat-avatar-mlx` package shape:

```bash
.venv-mlx/bin/pip install git+https://github.com/xocialize/longcat-avatar-mlx
```

System prerequisite:

- FFmpeg executable: required for audio/video handling and MP4 export. It must be
  visible on the PATH used by the runner environment.

macOS Homebrew example:

```bash
brew install ffmpeg
ffmpeg -version
```

## Nodes

| Node | Purpose |
| --- | --- |
| `LongCat Avatar MLX External Runner` | Launches the Apple MLX runner in an isolated subprocess and returns generated frames/video paths |
| `LongCat Avatar Whisper` | Loads Avatar 1.5 Whisper audio encoder assets for the legacy graph path |
| `LongCat Avatar Audio Crop` | Trims input audio and provides a cropped-audio preview |
| `LongCat Avatar Audio Encode` | Builds reusable audio conditioning for existing LongCat graph compatibility |
| `LongCat Avatar Audio Window` | Slices reusable audio conditioning for continuation windows |
| `LongCat Avatar Vocal Model` | Loads optional ONNX vocal separation model |
| `LongCat Avatar Vocal Extract` | Extracts vocals before audio conditioning |

The MLX node uses a JSON request/response contract and writes a retained
job-local `runner.log` when job retention is enabled. Runner logs are sanitized
and intended for diagnosing long-running Apple Silicon jobs.

## MLX Runner Smoke Gate

This branch includes a q4 support-gate smoke harness:

```bash
python scripts/run_mlx_q4_smoke_gate.py \
  --runner-python /path/to/.venv-mlx/bin/python \
  --weights-root /path/to/ComfyUI/models/longcat/LongCat-Video-Avatar-1.5-mlx \
  --image /path/to/reference.png \
  --audio /path/to/speech.wav \
  --prompt-file /path/to/prompt.txt \
  --output-root /path/to/ComfyUI/output \
  --evidence-json /path/to/evidence.json
```

The first support gate requires:

- `q4-merged` / q4 DMD merged MLX weights.
- 256 x 432 x 29 frame profile.
- macOS on Apple Silicon arm64.
- 32 GB+ unified memory evidence.
- Valid runner response JSON.
- A real MP4 or frame artifact.
- Content-free evidence; no prompts, private paths, credentials, cookies, or raw
  media payloads are accepted in support evidence.

Until that real artifact smoke passes on suitable hardware, this branch should
not claim public Apple Silicon support beyond repo-side implementation status.

## Reference Provenance

This branch uses the `longcat-avatar-mlx` project as the Apple MLX reference
port:

- Source: <https://github.com/xocialize/longcat-avatar-mlx>
- HF collection: <https://huggingface.co/collections/mlx-community/longcat-video-avatar-15-mlx-6a185d1af4a43074d882e375>

The official upstream model remains Meituan LongCat:

- Official LongCat-Video: <https://github.com/meituan-longcat/LongCat-Video>
- Official Avatar 1.5 weights: <https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5>

## License

This project is licensed under the [MIT License](LICENSE). The MLX reference
port is MIT licensed and adapted from work by the Meituan LongCat Team.
