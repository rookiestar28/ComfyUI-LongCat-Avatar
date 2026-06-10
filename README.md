# ComfyUI LongCat Avatar

This repository adapts the official Avatar 1.5 pipeline to ComfyUI. It focuses on CUDA inference, Whisper-large-v3 audio conditioning, required distill LoRA inference, single-audio and multi-audio avatar generation, single-file `.safetensors` DiT loading, official sharded DiT loading, official sharded INT8 DiT loading, and bounded automatic download of known official checkpoint assets.

<p align="center">
  <a href="https://github.com/meituan-longcat/LongCat-Video">Official LongCat-Video Repo</a> |
  <a href="https://meigen-ai.github.io/LongCat-Video-Avatar-1.5-Page/">Official Avatar 1.5 Project Page</a>
</p>

<div align="center">
  <video src="https://github.com/user-attachments/assets/218e76e1-0d82-4097-b880-5310b3bb1b0c" width="900"></video>
</div>

## Table of Contents

- [Feature Status](#feature-status)
- [Installation](#installation)
- [Required Model Files](#required-model-files)
- [Inference Weight Modes](#inference-weight-modes)
- [Model Source Boundaries](#model-source-boundaries)
- [Automatic Official Weight Download](#automatic-official-weight-download)
- [Nodes](#nodes)
- [Attention Backends](#attention-backends)
- [Audio Conditioning And Dual-Person Mode](#audio-conditioning-and-dual-person-mode)
- [Generation Controls](#generation-controls)
- [Avatar 1.5 Usage Tips](#avatar-15-usage-tips)
- [License](#license)

## Feature Status

| Capability | Status |
| --- | --- |
| Avatar 1.5 single audio `ai2v` | Supported |
| Avatar 1.5 single audio `at2v` | Supported by sampler mode; the current ComfyUI node still requires an image socket in the graph |
| Avatar 1.5 multi audio | Supported with `ai2v` only |
| 480p and 720p generation | Supported |
| Distill inference | Required for Avatar 1.5; sampler defaults are 8 steps, text CFG 1.0, audio CFG 1.0 |
| Single-file DiT `.safetensors` | Supported from `ComfyUI/models/diffusion_models/` |
| Official sharded DiT | Supported for Avatar 1.5 `base_model/` checkpoints |
| Official sharded INT8 DiT | Supported for Avatar 1.5 `base_model_int8/` checkpoints |
| Manifest-backed official weight download | Supported for known official Avatar 1.5 sharded assets |
| Selectable attention backends | Supported for `auto`, `sdpa`, `flash_attn_2`, `flash_attn_3`, `xformers`, `sageattn`, and `sageattn_3` |
| GGUF DiT | Not supported yet |
| Avatar 1.0 | Not supported by this ComfyUI contract |
| CPU or MPS inference | Not supported for the CUDA-only model path |

## Installation

Install the custom node inside ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rookiestar28/ComfyUI-LongCat-Avatar
cd ComfyUI-LongCat-Avatar
pip install -r requirements.txt
```

Use the Python environment that launches ComfyUI. The runtime path is CUDA-oriented and expects a compatible NVIDIA GPU plus a working CUDA PyTorch build already installed for ComfyUI. The default `requirements.txt` intentionally does not install `torch`, `torchvision`, `torchaudio`, FlashAttention, xFormers, SageAttention, Streamlit, OpenAI SDK, or upstream demo/server-only packages, because replacing those packages can break an existing ComfyUI setup.

macOS/MPS inference is not supported in this release. Any MPS work must happen on a dedicated experimental branch and must not be merged into the public CUDA path until Apple Silicon smoke inference proves model loading, attention fallback, audio encoding, VAE encode/decode, and end-to-end timing.

Optional dependency groups:

| File | Use When | Notes |
| --- | --- | --- |
| `requirements.txt` | Default node install | Required Python packages for normal model loading, audio encoding, and sampling. |
| `requirements-vocal.txt` | You use `LongCat Avatar Vocal Extract` | Installs `audio-separator` and ONNX Runtime dependencies. The node imports these only when the vocal extraction path is used. |
| `requirements-acceleration.txt` | You want optional attention acceleration | Documentation-only examples for FlashAttention, xFormers, and SageAttention. Install versions manually that match your ComfyUI PyTorch/CUDA stack. |

Example optional vocal install:

```bash
pip install -r requirements-vocal.txt
```

## Required Model Files

**Place model files in the normal ComfyUI model folders so the node dropdowns can find them:**

Some official assets can be downloaded automatically from inside the nodes. Enable `auto_download_missing_weights` on `(auto)Load LongCat Avatar Model` for official sharded Avatar 1.5 DiT checkpoints, and enable `auto_download_missing_text_encoder` on `LongCat Avatar Text Encode` for the official shared LongCat-Video tokenizer/text encoder. VAE, Whisper, and the selectable distill LoRA still need to be placed in the normal ComfyUI model folders.

- ComfyUI/models/diffusion_models/
  LongCat-Video-Avatar-1.5-int8.safetensors

- ComfyUI/models/loras/
  longcat-avatar-dmd_lora.safetensors

- ComfyUI/models/vae/
  LongCat-Video-Avatar-vae.safetensors

- ComfyUI/models/clip/
  umt5_xxl_fp8_e4m3fn_scaled.safetensors

- ComfyUI/models/audio_encoders/
  whisper-large-v3.safetensors

- ComfyUI/models/longcat/
  Kim_Vocal_2.onnx

- ComfyUI/models/longcat/LongCat-Video/
  tokenizer/
  text_encoder/

**Official sharded checkpoints use the `longcat` model folder:**

- ComfyUI/models/longcat/LongCat-Video-Avatar-1.5/base_model/
  config.json
  diffusion_pytorch_model.safetensors.index.json
  diffusion_pytorch_model-00001-of-00006.safetensors
  ...


- ComfyUI/models/longcat/LongCat-Video-Avatar-1.5/base_model_int8/
  config.json
  quantization_config.json
  quantized_model.safetensors.index.json
  quantized_model-00001-of-00004.safetensors
  ...

Download sources:

- Avatar 1.5 official weights: <https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5>
- Shared LongCat-Video tokenizer/text encoder: <https://huggingface.co/meituan-longcat/LongCat-Video>
- Community INT8 merged DiT: <https://huggingface.co/smthem/LongCat-Video-Avatar-1.5-merge>
- UMT5 text encoder: <https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders>

| Model file | Description | Download link | ComfyUI filename |
| --- | --- | --- | --- |
| Whisper-large-v3 | Avatar 1.5 audio encoder used by `LongCat Avatar Whisper` | [Hugging Face](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5/blob/main/whisper-large-v3/model.safetensors) | Download as `model.safetensors`, then rename to `whisper-large-v3.safetensors` under `ComfyUI/models/audio_encoders/`. |
| Distill LoRA | Required Avatar 1.5 distillation LoRA for 8-step inference | [Hugging Face](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5/blob/main/lora/dmd_lora.safetensors) | Keep as `dmd_lora.safetensors` or rename to `longcat-avatar-dmd_lora.safetensors` under `ComfyUI/models/loras/`. |

The VAE filename is selectable from the ComfyUI VAE dropdown; `LongCat-Video-Avatar-vae.safetensors` and `LongCat_Avatar_1.5_vae.safetensors` are both acceptable local filenames as long as the selected file is the Avatar 1.5 VAE.

Automatic download coverage:

- `auto_download_missing_weights` only applies to the official Avatar 1.5 sharded DiT checkpoint modes, described below.
- VAE is not included in the automatic download manifest. Keep `LongCat-Video-Avatar-vae.safetensors` in `ComfyUI/models/vae/`.
- The official manifest includes the upstream `lora/dmd_lora.safetensors` asset for checkpoint completeness, but the ComfyUI node still loads the selected distill LoRA from `ComfyUI/models/loras/`. Keep `longcat-avatar-dmd_lora.safetensors` in `ComfyUI/models/loras/` so the LoRA dropdown can find it.
- The official UMT5 text encoder is shared with the base `LongCat-Video` model. `LongCat Avatar Text Encode` can auto-download the bounded `tokenizer/*` and `text_encoder/*` assets from `meituan-longcat/LongCat-Video` into `ComfyUI/models/longcat/LongCat-Video/`. Existing workflows with `Load CLIP` connected to `LongCat Avatar Text Encode` use the single-file UMT5 fallback path.

## Inference Weight Modes

`(auto)Load LongCat Avatar Model` exposes three supported inference weight modes. The selected mode determines the DiT source automatically; there is no separate `diffusion_models` or `official_checkpoint` selector on the node.

| Mode | Auto-selected source | Expected files |
| --- | --- | --- |
| `single_file_safetensors` | `LongCat-Video-Avatar-1.5-int8.safetensors` | One `.safetensors` DiT file in `ComfyUI/models/diffusion_models/` |
| `official_sharded` | `LongCat-Video-Avatar-1.5/base_model/diffusion_pytorch_model.safetensors.index.json` | Official `base_model/` index and all referenced shards |
| `official_int8_sharded` | `LongCat-Video-Avatar-1.5/base_model_int8/quantized_model.safetensors.index.json` | Official `base_model_int8/` index, `quantization_config.json`, and all referenced shards |

Single-file mode loads the default merged/community DiT filename from `ComfyUI/models/diffusion_models/`. Official sharded modes validate the checkpoint index JSON, required config files, and every shard referenced by the index before inference begins.

GGUF remains unsupported and is not exposed as a model-node selector.

## Model Source Boundaries

The official sharded modes are the official checkpoint contract for this repository. They load the official Avatar 1.5 `base_model/` or `base_model_int8/` layouts from `ComfyUI/models/longcat/` and validate the official index files before tensor loading.

Single-file `.safetensors` mode is for local or community-converted DiT files in `ComfyUI/models/diffusion_models/`. A converted single-file checkpoint can originate from official weights, but it is not the official sharded layout and is not treated as an `official_sharded` or `official_int8_sharded` source.

Third-party wrapper-specific conversions and scheduler names are engineering references only. They do not redefine this repository's tensor schema, checkpoint layout, scheduler semantics, or supported runtime modes. This repository does not require third-party wrapper conversion before use.

FP16, FP8, GGUF, and alternate scheduler controls are not accepted official runtime modes in this release. They remain unsupported or experimental until implemented with explicit compatibility checks, tests, and documentation.

Official runtime precision remains bf16 by default. Official INT8 support is selected through `official_int8_sharded` as a checkpoint/weight mode, not through a generic FP8 runtime switch. FP16, FP8, and GGUF are not runtime precision modes in this release.

## Automatic Official Weight Download

`auto_download_missing_weights` is available as an on/off toggle on `(auto)Load LongCat Avatar Model`. When enabled with `official_sharded` or `official_int8_sharded`, the node builds a fixed manifest for `meituan-longcat/LongCat-Video-Avatar-1.5` and downloads missing official checkpoint assets under:

```text
ComfyUI/models/longcat/LongCat-Video-Avatar-1.5/
```

The downloader is intentionally bounded:

- It is manifest-backed and limited to known official assets.
- It does not accept arbitrary URLs.
- It does not accept arbitrary repo IDs.
- It does not add a token text box; use normal Hugging Face local authentication when your environment requires gated access.
- It does not replace every ComfyUI model dropdown. VAE, Whisper, and the distill LoRA still need to be present in their normal ComfyUI model folders.
- It does not download the VAE into `ComfyUI/models/vae/`.
- It does not register the official repo's `lora/dmd_lora.safetensors` as a selectable LoRA in `ComfyUI/models/loras/`; place or copy the required distill LoRA there before loading the model node.

`auto_download_missing_text_encoder` is available on `LongCat Avatar Text Encode`. When enabled, it downloads only the shared base `meituan-longcat/LongCat-Video` `tokenizer/*` and `text_encoder/*` assets into:

```text
ComfyUI/models/longcat/LongCat-Video/
```

`LongCat Avatar Text Encode` also exposes its own `offload_device` control for the native official path. The default is `cpu` so the native UMT5 text encoder weights do not consume VRAM; choose `cuda` only when you have enough headroom and want faster text encoding. If the `clip` input is connected, the node uses the existing ComfyUI `Load CLIP` single-file UMT5 fallback and does not load the native text encoder.

Manual equivalent:

```bash
hf download meituan-longcat/LongCat-Video-Avatar-1.5 --local-dir ComfyUI/models/longcat/LongCat-Video-Avatar-1.5
```

## Nodes

| Node | Purpose |
| --- | --- |
| `(auto)Load LongCat Avatar Model` | Loads the DiT source selected by `inference_weight_mode`, VAE, and required Avatar 1.5 distill LoRA |
| `LongCat Avatar Whisper` | Loads `whisper-large-v3.safetensors` from `ComfyUI/models/audio_encoders/` |
| `LongCat Avatar Text Encode` | Preferred official-first text node. Without `clip`, it uses the shared official `LongCat-Video` tokenizer and UMT5 text encoder; with `clip`, it uses the ComfyUI `Load CLIP` single-file fallback |
| `LongCat Avatar Audio Crop` | Trims a ComfyUI audio input by start/end time and provides an inline cropped-audio preview |
| `LongCat Avatar Audio Encode` | Builds reusable full-clip single-audio or optional dual-audio conditioning |
| `LongCat Avatar Audio Window` | Optionally slices reusable full-clip audio conditioning for explicit continuation windows |
| `LongCat Avatar Sampler` | Generates frames in `ai2v` or `at2v` mode and can optionally mux audio into a saved video |
| `LongCat Avatar Vocal Model` | Loads an optional ONNX vocal separation model from `ComfyUI/models/longcat/` |
| `LongCat Avatar Vocal Extract` | Extracts vocals from an input audio clip before audio conditioning |

Typical graph order:

1. Load the model with `(auto)Load LongCat Avatar Model`.
2. Prefer `LongCat Avatar Text Encode` without connecting `clip` for official prompt-conditioning parity. It can auto-download the shared base `LongCat-Video` tokenizer/text encoder assets when `auto_download_missing_text_encoder` is enabled.
3. Load Whisper with `LongCat Avatar Whisper`.
4. Existing `Load CLIP` + `LongCat Avatar Text Encode` graphs remain supported as the UMT5 single-file fallback path.
5. Optionally trim source audio with `LongCat Avatar Audio Crop`.
6. Encode audio with `LongCat Avatar Audio Encode`.
7. Generate with `LongCat Avatar Sampler`.

## Attention Backends

`(auto)Load LongCat Avatar Model` includes an `attention_mode` selector:

| Mode | Behavior |
| --- | --- |
| `auto` | Preserves the attention flags from the selected LongCat checkpoint config. Official configs currently default to FlashAttention 2. If an optional backend from the config is unavailable, the runtime prints a warning and falls back to SDPA where supported. |
| `sdpa` | Forces PyTorch `scaled_dot_product_attention`. This is the most portable fallback. |
| `flash_attn_2` | Forces FlashAttention 2. The loader raises a clear error if the optional package is unavailable in the ComfyUI Python environment. |
| `flash_attn_3` | Forces FlashAttention 3. The loader raises a clear error if the optional package is unavailable in the ComfyUI Python environment. |
| `xformers` | Forces xFormers memory-efficient attention. The loader raises a clear error if xFormers is unavailable. |
| `sageattn` | Forces SageAttention for DiT attention. The loader raises a clear error if SageAttention is unavailable. |
| `sageattn_3` | Reserved for SageAttention3 Blackwell kernels. This mode must be fully available and compatible; otherwise the loader raises a clear error. |

SageAttention support is optional. Install and validate the package in the same Python environment that launches ComfyUI. The node uses lazy imports and does not require SageAttention for startup or CPU-only tests. At model load, the console prints the requested `attention_mode`, effective backend flags, and backend availability so slow SDPA fallback is visible.

## Audio Conditioning And Dual-Person Mode

`LongCat Avatar Audio Encode` supports a single-person path and an optional two-person dialogue path.

For single-person generation, connect only `audio`. Leave `left_audio` disconnected.

For two-person dialogue generation, the audio inputs are assigned by the characters' visual positions in the input image or video frame:

| Input | Meaning |
| --- | --- |
| `left_audio` | Speech for the person on the left side of the image or video frame |
| `audio` | Speech for the person on the right side of the image or video frame |

The left/right naming does not refer to stereo audio channels. It refers to where the people appear in the visual frame. In a two-person image, connect the left person's speech to `left_audio` and the right person's speech to `audio`.

`audio_type` controls how the two clips are combined:

| `audio_type` | Meaning |
| --- | --- |
| `para` | Parallel dialogue. The left and right clips are treated as simultaneous tracks and should have matching duration. |
| `add` | Sequential dialogue. The left clip is placed before the right clip with silence-padding semantics. |

`p_box` optionally tells the node where the people are in the image. Use comma-separated bounding boxes in this order:

```text
[left_y_min, left_x_min, left_y_max, left_x_max], [right_y_min, right_x_min, right_y_max, right_x_max]
```

Measure these coordinates on the original image connected to `Load Image`, not on the generated 480p/720p output. The pipeline builds masks from the original-image coordinates first, then resizes and center-crops the masks together with the image for inference.

To get the values, open the source image in an image editor or viewer that shows cursor coordinates, then record each person's bounding rectangle:

1. Read the top-left point as `x_min`, `y_min`.
2. Read the bottom-right point as `x_max`, `y_max`.
3. Enter the box in LongCat order: `[y_min, x_min, y_max, x_max]`.

Example: if the left person covers `x=120..430`, `y=80..760`, and the right person covers `x=520..820`, `y=90..770`, enter:

```text
[80, 120, 760, 430], [90, 520, 770, 820]
```

The first box maps to `left_audio`; the second box maps to `audio`. A third box may be provided for other/background people that should receive silent background conditioning:

```text
[left_y_min, left_x_min, left_y_max, left_x_max], [right_y_min, right_x_min, right_y_max, right_x_max], [other_y_min, other_x_min, other_y_max, other_x_max]
```

If `p_box` is empty, the node still builds audio conditioning, but it does not provide explicit person-location boxes to the multi-person path.

## Generation Controls

- `resolution`: `480p` maps to 480x832 and `720p` maps to 768x1280.
- `steps`: defaults to 8 for Avatar 1.5 DMD/distill inference; accepted range is 1 to 50. When the DMD/distill LoRA is active, the effective value is fixed to the official 8-step contract.
- `text_guidance_scale`: defaults to 1.0; accepted range is 1.0 to 10.0. It only affects non-DMD normal inference; Avatar 1.5 DMD/distill inference forces the effective text CFG to 1.0.
- `audio_guidance_scale`: defaults to 1.0; accepted range is 1.0 to 10.0. It only affects non-DMD normal inference; Avatar 1.5 DMD/distill inference forces the effective audio CFG to 1.0.
- `LongCat Avatar Audio Encode` computes `num_segments` automatically from the prepared input audio length and current `save_fps`. The first generated window is 93 frames; each continuation adds 80 new frames because the official window keeps 13 overlap frames. At 25 fps, an 18 second clip computes to 6 segments so generation covers the full audio. The sampler trims the generated segment envelope back to the source-audio frame count before returning image frames or muxing video.
- `save_fps`: defaults to 25. Treat 25 fps as the reliable Avatar 1.5 setting. Changing this value also changes audio-window segmentation and target output frame counts, but local validation showed non-25 fps values can make lip sync and motion timing unstable.
- `ref_img_index`: integer frame index for the reference image insertion/attention anchor used during continuation. It defaults to 10; official guidance commonly uses values in the 0 to 24 range for consistency.
- `mask_frame_range`: integer radius around `ref_img_index` where reference-frame attention is masked to reduce repeated motion. It defaults to 3; larger values may reduce repeated actions but can introduce artifacts.
- `p_box`: optional person boxes for `LongCat Avatar Audio Encode`; see [Audio Conditioning And Dual-Person Mode](#audio-conditioning-and-dual-person-mode) for left/right audio mapping and coordinate order.
- `block_num`: integer memory-streaming control. `0` uses eager full DiT loading on CUDA and offloads the DiT back to CPU after generation. Values from `1` to `64` enable streaming prefetch mode with that block count.
- `offload_device`: VAE offload target used during denoising. `cpu` is the default and preserves the lower-VRAM behavior. `cuda` keeps the VAE resident on the active CUDA device to avoid VAE CPU/GPU transfers, but requires more VRAM.
- `inference_weight_mode`: choose `single_file_safetensors`, `official_sharded`, or `official_int8_sharded`; the node auto-selects the corresponding default DiT source.
- `auto_download_missing_weights`: on/off toggle for bounded official sharded checkpoint download when required files are missing.
- `auto_download_missing_text_encoder`: on/off toggle on text encode nodes for bounded shared base `LongCat-Video` tokenizer/text encoder download.
- `LongCat Avatar Text Encode` `offload_device`: chooses where the native official UMT5 text encoder is loaded for prompt encoding. `cpu` is the default to avoid loading the large text encoder into VRAM; `cuda` can be faster but needs substantially more VRAM. This is ignored when `clip` is connected for fallback.
- `mux_audio_path`: optional local audio file path used to mux audio into the generated video. Leave it empty to return only image frames and an empty `video_path`. When provided, the node saves a muxed `.mp4` under the ComfyUI output directory using the built-in `longcat_avatar` filename prefix.

Alternate scheduler controls are not exposed. Reference-wrapper scheduler names such as `longcat_distill_euler` are not adopted unless official Avatar 1.5 parity is proven with source comparison and tests.

## Avatar 1.5 Usage Tips

| Tip | Repo-aligned guidance |
| --- | --- |
| Prompt detail | Use longer, descriptive prompts for better consistency. Include character appearance, speaking action, clothing, and scene context, for example: `A young woman with long black hair is speaking and smiling, wearing a white blouse, sitting in a bright cafe.` |
| Reference frame controls | `ref_img_index` defaults to 10. Values in the 0 to 24 range are commonly used for consistency; 30 can be tried when repeated motion is visible. `mask_frame_range` defaults to 3; larger values may reduce repeated actions but can introduce artifacts. |
| Resolution | The sampler supports `480p` and `720p` through the `resolution` control. In this ComfyUI node, `480p` maps to 480x832 and `720p` maps to 768x1280. |
| Dual-audio mode | `audio_type=para` is for simultaneous two-person dialogue and requires equal prepared clip lengths. `audio_type=add` makes a sequential turn-taking track: `left_audio` first, then `audio`, with silence padding semantics. |
| Avatar version | This ComfyUI package supports Avatar 1.5 only. Avatar 1.5 uses Whisper-large-v3 audio conditioning; Avatar 1.0/Wav2Vec2 is not a supported runtime path in this repo. |
| Distill inference | Avatar 1.5 requires the DMD/distill LoRA. When the DMD LoRA is active, the effective sampler values are fixed to 8 steps, text CFG 1.0, and audio CFG 1.0; UI CFG inputs are accepted but do not affect inference. CFG tuning only applies to non-DMD normal inference. |
| INT8 mode | INT8 is supported for Avatar 1.5 through `official_int8_sharded` or a compatible INT8 single-file DiT source. Use it when you want lower VRAM usage and have the required INT8 files available. |

## License

This project is licensed under the [MIT License](LICENSE).
