import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.support.comfy_stubs import FakeSchema, loaded_longcat_extension_module, loaded_longcat_node_module


def port_by_name(schema: FakeSchema, name: str):
    for port in schema.inputs:
        if port.name == name:
            return port
    raise AssertionError(f"Missing input port: {name}")


class FakeTextTensor:
    shape = (1, 512, 4096)

    def repeat(self, *args):
        return self

    def view(self, *shape):
        self.shape = tuple(4096 if dim == -1 else dim for dim in shape)
        return self

    def to(self, *args):
        return self


class FakeClip:
    def tokenize(self, text):
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens):
        return [[FakeTextTensor()]]


class FakeAudioWaveform:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __getitem__(self, key):
        sample_slice = key[-1] if isinstance(key, tuple) else key
        start = 0 if sample_slice.start is None else sample_slice.start
        stop = self.shape[-1] if sample_slice.stop is None else sample_slice.stop
        return FakeAudioWaveform((*self.shape[:-1], max(0, stop - start)))


class NodeSchemaContractTests(unittest.TestCase):
    def test_all_node_schemas_build_with_cpu_only_stubs(self):
        with loaded_longcat_node_module() as node_module:
            node_classes = [
                node_module.LongCat_Video_SM_Model,
                node_module.LongCat_Video_SM_Sampler,
                node_module.LongCat_Video_SM_Encode,
                node_module.LongCat_Video_SM_Audio,
                node_module.LongCat_Video_SM_AudioWindow,
                node_module.LongCat_Video_SM_AudioCrop,
                node_module.LongCat_Video_SM_MLXGenerate,
                node_module.LongCat_Video_SM_Vocal,
                node_module.LongCat_Video_SM_WhisperModel,
                node_module.LongCat_Video_SM_VocalModel,
            ]

            schemas = [node_class.define_schema() for node_class in node_classes]

        self.assertTrue(all(isinstance(schema, FakeSchema) for schema in schemas))
        self.assertEqual(
            [schema.node_id for schema in schemas],
            [
                "LongCat_Video_SM_Model",
                "LongCat_Video_SM_Sampler",
                "LongCat_Video_SM_Encode",
                "LongCat_Video_SM_Audio",
                "LongCat_Video_SM_AudioWindow",
                "LongCat_Video_SM_AudioCrop",
                "LongCat_Video_SM_MLXGenerate",
                "LongCat_Video_SM_Vocal",
                "LongCat_Video_SM_WhisperModel",
                "LongCat_Video_SM_VocalModel",
            ],
        )

    def test_model_schema_uses_autoload_controls(self):
        with loaded_longcat_node_module() as node_module:
            schema = node_module.LongCat_Video_SM_Model.define_schema()

        self.assertEqual(schema.display_name, "(auto)Load LongCat Avatar Model")
        self.assertEqual(schema.category, "LongCat Avatar")
        self.assertEqual(
            [port.name for port in schema.inputs],
            [
                "inference_weight_mode",
                "attention_mode",
                "auto_download_missing_weights",
                "vae",
                "lora",
            ],
        )
        self.assertEqual(
            port_by_name(schema, "inference_weight_mode").options,
            [
                "single_file_safetensors",
                "official_sharded",
                "official_int8_sharded",
            ],
        )
        self.assertEqual(
            list(port_by_name(schema, "attention_mode").options),
            ["sdpa"],
        )
        self.assertEqual(port_by_name(schema, "auto_download_missing_weights").port_type, "BOOLEAN")
        self.assertIs(port_by_name(schema, "auto_download_missing_weights").default, True)
        self.assertIn("LongCat-Video-Avatar-vae.safetensors", port_by_name(schema, "vae").options)
        self.assertIn("longcat-avatar-dmd_lora.safetensors", port_by_name(schema, "lora").options)
        self.assertEqual(schema.outputs[0].display_name, "model")

    def test_text_encode_schema_is_official_first_with_clip_fallback(self):
        with loaded_longcat_node_module() as node_module:
            schema = node_module.LongCat_Video_SM_Encode.define_schema()

        self.assertEqual(schema.display_name, "LongCat Avatar Text Encode")
        self.assertEqual(
            [port.name for port in schema.inputs],
            [
                "clip",
                "text_encoder_root",
                "auto_download_missing_text_encoder",
                "offload_device",
                "prompt",
                "negative_prompt",
            ],
        )
        self.assertTrue(port_by_name(schema, "clip").optional)
        self.assertEqual(port_by_name(schema, "text_encoder_root").default, "LongCat-Video")
        self.assertEqual(port_by_name(schema, "auto_download_missing_text_encoder").port_type, "BOOLEAN")
        self.assertIs(port_by_name(schema, "auto_download_missing_text_encoder").default, True)
        self.assertEqual(port_by_name(schema, "offload_device").options, ["cpu"])
        self.assertEqual(port_by_name(schema, "offload_device").default, "cpu")

    def test_text_encode_execute_uses_official_path_when_clip_is_missing(self):
        with loaded_longcat_node_module() as node_module:
            calls = []
            expected = {"te": "official"}
            node_module.clear_comfyui_cache = lambda: None
            node_module.resolve_or_download_official_text_encoder_layout = lambda *args, **kwargs: "layout"

            def fake_encode(**kwargs):
                calls.append(kwargs)
                return expected

            node_module.encode_official_text_conditioning = fake_encode

            result = node_module.LongCat_Video_SM_Encode.execute(
                clip=None,
                text_encoder_root="LongCat-Video",
                auto_download_missing_text_encoder=True,
                offload_device="cpu",
                prompt="prompt",
                negative_prompt="negative",
            )

        self.assertEqual(result, (expected,))
        self.assertEqual(calls[0]["layout"], "layout")
        self.assertEqual(calls[0]["offload_device"], "cpu")
        self.assertEqual(calls[0]["dtype"], "bfloat16")

    def test_text_encode_execute_uses_mps_auto_fp16_policy(self):
        with loaded_longcat_node_module() as node_module:
            calls = []
            expected = {"te": "official"}
            node_module.clear_comfyui_cache = lambda: None
            node_module.get_runtime_device = lambda: "mps"
            node_module.resolve_or_download_official_text_encoder_layout = lambda *args, **kwargs: "layout"

            def fake_encode(**kwargs):
                calls.append(kwargs)
                return expected

            node_module.encode_official_text_conditioning = fake_encode

            result = node_module.LongCat_Video_SM_Encode.execute(
                clip=None,
                text_encoder_root="LongCat-Video",
                auto_download_missing_text_encoder=True,
                offload_device="cpu",
                prompt="prompt",
                negative_prompt="negative",
            )

        self.assertEqual(result, (expected,))
        self.assertEqual(calls[0]["device"], "mps")
        self.assertEqual(calls[0]["dtype"], "float16")

    def test_text_encode_execute_uses_clip_fallback_when_clip_is_connected(self):
        with loaded_longcat_node_module() as node_module:
            node_module.clear_comfyui_cache = lambda: None

            def fail_if_called(*args, **kwargs):
                raise AssertionError("connected clip must select fallback path")

            node_module.resolve_or_download_official_text_encoder_layout = fail_if_called
            node_module.encode_official_text_conditioning = fail_if_called

            result = node_module.LongCat_Video_SM_Encode.execute(
                clip=FakeClip(),
                text_encoder_root="LongCat-Video",
                auto_download_missing_text_encoder=True,
                offload_device="cpu",
                prompt="prompt",
                negative_prompt="negative",
            )

        self.assertEqual(result[0]["conditioning_source"], "comfy_clip_umt5")

    def test_no_duplicate_official_text_encode_node_is_registered(self):
        with loaded_longcat_node_module() as node_module:
            self.assertFalse(hasattr(node_module, "LongCat_Video_SM_OfficialTextEncode"))

    def test_model_schema_does_not_expose_raw_download_inputs(self):
        with loaded_longcat_node_module() as node_module:
            schema = node_module.LongCat_Video_SM_Model.define_schema()

        forbidden_names = {
            "repo_id",
            "download_url",
            "hf_token",
            "model_json",
            "checkpoint_path",
            "diffusion_models",
            "official_checkpoint",
            "gguf_experimental_unsupported",
        }
        self.assertTrue(forbidden_names.isdisjoint({port.name for port in schema.inputs}))

    def test_model_execute_rejects_mps_auto_attention_before_loading(self):
        with loaded_longcat_node_module() as node_module:
            node_module.get_runtime_device = lambda: "mps"

            with self.assertRaisesRegex(RuntimeError, "limited to explicit 'sdpa'"):
                node_module.LongCat_Video_SM_Model.execute(
                    inference_weight_mode="single_file_safetensors",
                    attention_mode="auto",
                    auto_download_missing_weights=True,
                    vae="none",
                    lora="none",
                )

    def test_model_execute_passes_mps_dtype_policy_to_loader(self):
        with loaded_longcat_node_module() as node_module:
            calls = []
            policy_calls = []
            node_module.get_runtime_device = lambda: "mps"

            policy = SimpleNamespace(
                backend="mps",
                text_encoder_precision="bf16",
                audio_encoder_precision="fp32",
                dit_precision="bf16",
                vae_precision="bf16",
                math_precision="fp32",
                text_encoder_dtype="bfloat16",
                vae_dtype="bfloat16",
                dit_dtype="bfloat16",
            )

            def fake_policy(device, **kwargs):
                policy_calls.append((device, kwargs))
                return policy

            def fake_load(*args, **kwargs):
                calls.append((args, kwargs))
                return "model"

            node_module.resolve_backend_dtype_policy = fake_policy
            node_module.load_longcat_video_model = fake_load

            result = node_module.LongCat_Video_SM_Model.execute(
                inference_weight_mode="single_file_safetensors",
                attention_mode="sdpa",
                auto_download_missing_weights=True,
                vae="none",
                lora="none",
            )

        self.assertEqual(result, ("model",))
        self.assertEqual(policy_calls[0][0], "mps")
        self.assertEqual(calls[0][1]["tokenizer_dtype"], "bfloat16")
        self.assertEqual(calls[0][1]["vae_dtype"], "bfloat16")
        self.assertEqual(calls[0][1]["scheduler_dtype"], "bfloat16")
        self.assertEqual(calls[0][1]["dit_dtype"], "bfloat16")

    def test_model_loader_exposes_component_dtype_parameters(self):
        source = Path("LongCat_Video/run_demo_avatar_single_audio_to_video.py").read_text(encoding="utf-8")

        self.assertIn("tokenizer_dtype=None", source)
        self.assertIn("vae_dtype=None", source)
        self.assertIn("scheduler_dtype=None", source)
        self.assertIn("dit_dtype=None", source)
        self.assertIn("torch_dtype=tokenizer_dtype", source)
        self.assertIn("torch_dtype=vae_dtype", source)
        self.assertIn("vae=vae.eval().to(vae_dtype)", source)
        self.assertIn("torch_dtype=scheduler_dtype", source)
        self.assertIn("dit=dit.eval().to(dit_dtype)", source)

    def test_model_mode_resolves_autoload_sources(self):
        with loaded_longcat_node_module() as node_module:
            self.assertEqual(
                node_module._resolve_single_file_model_name("single_file_safetensors"),
                "LongCat-Video-Avatar-1.5-int8.safetensors",
            )
            self.assertIsNone(node_module._resolve_single_file_model_name("official_sharded"))
            self.assertEqual(
                node_module._resolve_official_checkpoint_name("official_sharded"),
                "LongCat-Video-Avatar-1.5/base_model/diffusion_pytorch_model.safetensors.index.json",
            )
            self.assertEqual(
                node_module._resolve_official_checkpoint_name("official_int8_sharded"),
                "LongCat-Video-Avatar-1.5/base_model_int8/quantized_model.safetensors.index.json",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported inference_weight_mode"):
                node_module._resolve_single_file_model_name("bad_mode")

    def test_sampler_schema_exposes_official_distill_and_video_defaults(self):
        with loaded_longcat_node_module() as node_module:
            schema = node_module.LongCat_Video_SM_Sampler.define_schema()

        self.assertEqual(schema.display_name, "LongCat Avatar Sampler")
        self.assertEqual(schema.category, "LongCat Avatar")
        self.assertEqual(port_by_name(schema, "steps").default, 8)
        self.assertEqual(port_by_name(schema, "steps").min, 1)
        self.assertEqual(port_by_name(schema, "steps").max, 50)
        self.assertEqual(port_by_name(schema, "text_guidance_scale").default, 1.0)
        self.assertEqual(port_by_name(schema, "text_guidance_scale").min, 1.0)
        self.assertEqual(port_by_name(schema, "text_guidance_scale").max, 10.0)
        self.assertEqual(port_by_name(schema, "audio_guidance_scale").default, 1.0)
        self.assertEqual(port_by_name(schema, "audio_guidance_scale").min, 1.0)
        self.assertEqual(port_by_name(schema, "audio_guidance_scale").max, 10.0)
        self.assertEqual(port_by_name(schema, "block_num").default, 1)
        self.assertEqual(port_by_name(schema, "block_num").min, 0)
        self.assertEqual(port_by_name(schema, "block_num").max, 64)
        self.assertEqual(port_by_name(schema, "mux_audio_path").default, "")
        self.assertEqual(port_by_name(schema, "offload_device").options, ["cpu"])
        self.assertEqual(port_by_name(schema, "offload_device").default, "cpu")
        self.assertEqual(port_by_name(schema, "debug_mode").default, False)
        self.assertNotIn("save_video", {port.name for port in schema.inputs})
        self.assertNotIn("video_prefix", {port.name for port in schema.inputs})
        self.assertEqual([port.display_name for port in schema.outputs], ["image", "video_path"])

    def test_sampler_mux_audio_path_controls_video_output_without_prefix_input(self):
        with loaded_longcat_node_module() as node_module:
            calls = []
            runtime_plan_calls = []
            model = type("Model", (), {"use_distill": True})()
            plan = object()

            node_module.clear_comfyui_cache = lambda: None
            node_module.validate_text_conditioning_payload = lambda *args, **kwargs: None
            node_module.validate_audio_conditioning_payload = lambda *args, **kwargs: None

            def fake_build_sampler_execution_request(*args, **kwargs):
                mux_audio_path = "" if str(kwargs["mux_audio_path"]).strip() in ("", "0") else kwargs["mux_audio_path"]
                return SimpleNamespace(
                    mode="single",
                    stage_1=kwargs["stage_1"],
                    resolution=kwargs["resolution"],
                    seed=kwargs["seed"],
                    steps=kwargs["steps"],
                    text_guidance_scale=kwargs["text_guidance_scale"],
                    audio_guidance_scale=kwargs["audio_guidance_scale"],
                    ref_img_index=kwargs["ref_img_index"],
                    mask_frame_range=kwargs["mask_frame_range"],
                    block_num=kwargs["block_num"],
                    mux_audio_path=mux_audio_path,
                    offload_device=kwargs["offload_device"],
                )

            node_module.build_sampler_execution_request = fake_build_sampler_execution_request
            def fake_build_runtime_plan(*args, **kwargs):
                runtime_plan_calls.append((args, kwargs))
                return plan

            node_module.build_runtime_plan = fake_build_runtime_plan
            node_module.apply_runtime_plan = lambda *args, **kwargs: None
            node_module.cleanup_runtime_plan = lambda *args, **kwargs: None
            node_module.tensor2image = lambda value: value
            node_module.generate = lambda *args, **kwargs: "frames"

            def fake_save_muxed_video(*args, **kwargs):
                calls.append((args, kwargs))
                return "video.mp4" if kwargs["enabled"] else ""

            node_module.save_muxed_video = fake_save_muxed_video

            disabled = node_module.LongCat_Video_SM_Sampler.execute(
                model,
                "te",
                "au",
                "image",
                "ai2v",
                "480p",
                1,
                8,
                1.0,
                1.0,
                10,
                3,
                1,
                "",
                "cpu",
            )
            disabled_zero = node_module.LongCat_Video_SM_Sampler.execute(
                model,
                "te",
                "au",
                "image",
                "ai2v",
                "480p",
                1,
                8,
                1.0,
                1.0,
                10,
                3,
                1,
                "0",
                "cpu",
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = os.path.join(temp_dir, "speech.wav")
                Path(audio_path).write_bytes(b"fake")
                enabled = node_module.LongCat_Video_SM_Sampler.execute(
                    model,
                    "te",
                    "au",
                    "image",
                    "ai2v",
                    "480p",
                    1,
                    8,
                    1.0,
                    1.0,
                    10,
                    3,
                    1,
                    audio_path,
                    "cuda",
                )

        self.assertEqual(disabled, ("frames", ""))
        self.assertEqual(disabled_zero, ("frames", ""))
        self.assertEqual(enabled, ("frames", "video.mp4"))
        self.assertEqual([call[0][2] for call in runtime_plan_calls], ["cpu", "cpu", "cuda"])
        self.assertFalse(calls[0][1]["enabled"])
        self.assertEqual(calls[0][1]["audio_path"], "")
        self.assertIsNone(calls[0][1]["prefix"])
        self.assertFalse(calls[1][1]["enabled"])
        self.assertEqual(calls[1][1]["audio_path"], "")
        self.assertIsNone(calls[1][1]["prefix"])
        self.assertTrue(calls[2][1]["enabled"])
        self.assertTrue(calls[2][1]["audio_path"].endswith("speech.wav"))
        self.assertIsNone(calls[2][1]["prefix"])

    def test_sampler_rejects_invalid_mux_audio_path_before_generation(self):
        with loaded_longcat_node_module() as node_module:
            model = type("Model", (), {"use_distill": True})()
            calls = []

            node_module.clear_comfyui_cache = lambda: None
            node_module.validate_text_conditioning_payload = lambda *args, **kwargs: calls.append("text")
            node_module.validate_audio_conditioning_payload = lambda *args, **kwargs: calls.append("audio")

            def fail_if_called(*args, **kwargs):
                raise AssertionError("inference setup should not run for invalid mux_audio_path")

            node_module.build_sampler_execution_request = lambda *args, **kwargs: SimpleNamespace(
                mux_audio_path=kwargs["mux_audio_path"],
            )
            node_module.build_runtime_plan = fail_if_called
            node_module.apply_runtime_plan = fail_if_called
            node_module.generate = fail_if_called
            node_module.save_muxed_video = fail_if_called

            with self.assertRaisesRegex(FileNotFoundError, "mux_audio_path"):
                node_module.LongCat_Video_SM_Sampler.execute(
                    model,
                    "te",
                    "au",
                    "image",
                    "ai2v",
                    "480p",
                    1,
                    8,
                    1.0,
                    1.0,
                    10,
                    3,
                    1,
                    "missing.wav",
                )

        self.assertEqual(calls, [])

    def test_audio_schema_keeps_bbox_as_optional_text_control(self):
        with loaded_longcat_node_module() as node_module:
            schema = node_module.LongCat_Video_SM_Audio.define_schema()

        self.assertEqual(schema.display_name, "LongCat Avatar Audio Encode")
        self.assertNotIn("num_segments", {port.name for port in schema.inputs})
        self.assertEqual(port_by_name(schema, "save_fps").default, 25)
        self.assertEqual(port_by_name(schema, "audio_type").options, ["para", "add"])
        self.assertEqual(port_by_name(schema, "p_box").default, "")
        self.assertFalse(port_by_name(schema, "p_box").multiline)
        self.assertTrue(port_by_name(schema, "left_audio").optional)

    def test_audio_execute_ignores_legacy_num_segments_input(self):
        with loaded_longcat_node_module() as node_module:
            calls = []
            node_module.clear_comfyui_cache = lambda: None
            node_module.parse_person_boxes = lambda value: "boxes"

            def fake_get_audio_emb(*args, **kwargs):
                calls.append((args, kwargs))
                return {"num_segments": 6}

            node_module.get_audio_emb = fake_get_audio_emb
            result = node_module.LongCat_Video_SM_Audio.execute(
                "encoder",
                "audio",
                25,
                "para",
                "",
                left_audio=None,
                num_segments=3,
            )

        self.assertEqual(result, ({"num_segments": 6},))
        self.assertEqual(calls[0][0], ("encoder", "audio", None, "para", 25, "cuda:0"))
        self.assertEqual(calls[0][1], {"p_box": "boxes"})

    def test_audio_window_schema_exposes_continuation_controls(self):
        with loaded_longcat_node_module() as node_module:
            schema = node_module.LongCat_Video_SM_AudioWindow.define_schema()

        self.assertEqual(schema.display_name, "LongCat Avatar Audio Window")
        self.assertEqual(port_by_name(schema, "frames_processed").default, 0)
        self.assertEqual(port_by_name(schema, "num_frames").default, 93)
        self.assertEqual(port_by_name(schema, "overlap").default, 13)
        self.assertEqual(port_by_name(schema, "if_not_enough_audio").options, ["clamp", "mirror_from_end"])
        self.assertEqual([port.display_name for port in schema.outputs], ["au_cond_window"])

    def test_audio_crop_schema_exposes_time_controls_and_audio_output(self):
        with loaded_longcat_node_module() as node_module:
            schema = node_module.LongCat_Video_SM_AudioCrop.define_schema()

        self.assertEqual(schema.display_name, "LongCat Avatar Audio Crop")
        self.assertEqual(
            [port.name for port in schema.inputs],
            ["audio", "start_time", "end_time"],
        )
        self.assertEqual(port_by_name(schema, "audio").port_type, "AUDIO")
        self.assertEqual(port_by_name(schema, "start_time").default, "0:00")
        self.assertFalse(port_by_name(schema, "start_time").multiline)
        self.assertEqual(port_by_name(schema, "end_time").default, "1:00")
        self.assertEqual([port.display_name for port in schema.outputs], ["audio"])
        self.assertTrue(schema.is_output_node)
        self.assertEqual(schema.hidden, ["PROMPT", "EXTRA_PNGINFO"])

    def test_audio_crop_execute_returns_cropped_audio_and_preview_ui(self):
        with loaded_longcat_node_module() as node_module:
            audio = {"waveform": FakeAudioWaveform((1, 1, 5000)), "sample_rate": 100}

            result = node_module.LongCat_Video_SM_AudioCrop.execute(audio, "0:10", "0:30")

        self.assertEqual(result[0]["sample_rate"], 100)
        self.assertEqual(result[0]["waveform"].shape, (1, 1, 2000))
        self.assertIsNotNone(result.ui)
        self.assertEqual(result.ui.audio, result[0])
        self.assertIs(result.ui.cls, node_module.LongCat_Video_SM_AudioCrop)

    def test_extension_entrypoint_registers_audio_crop_node(self):
        with loaded_longcat_extension_module() as extension_module:
            extension = asyncio.run(extension_module.comfy_entrypoint())
            node_list = asyncio.run(extension.get_node_list())

        self.assertIn(
            "LongCat_Video_SM_AudioCrop",
            [node_class.__name__ for node_class in node_list],
        )
        self.assertIn(
            "LongCat_Video_SM_MLXGenerate",
            [node_class.__name__ for node_class in node_list],
        )

    def test_mlx_bridge_schema_is_external_runner_specific(self):
        with loaded_longcat_node_module() as node_module:
            schema = node_module.LongCat_Video_SM_MLXGenerate.define_schema()

        self.assertEqual(schema.display_name, "LongCat Avatar MLX External Runner")
        self.assertEqual(schema.category, "LongCat Avatar/MLX")
        self.assertEqual(
            [port.name for port in schema.inputs],
            [
                "image",
                "audio",
                "runner_python",
                "weights_root",
                "variant",
                "mode",
                "prompt",
                "negative_prompt",
                "height",
                "width",
                "num_frames",
                "fps",
                "seed",
                "timeout_seconds",
                "output_basename",
                "retain_job_dir",
            ],
        )
        self.assertEqual(port_by_name(schema, "variant").options, ["q4-merged", "q8-merged", "merged"])
        self.assertEqual(port_by_name(schema, "mode").options, ["dry-run", "generate"])
        self.assertNotIn("mps", schema.display_name.lower())
        self.assertEqual(
            [port.display_name for port in schema.outputs],
            ["video_path", "frames_path", "response_path", "job_dir"],
        )

    def test_mlx_bridge_execute_delegates_to_bridge_helper(self):
        with loaded_longcat_node_module() as node_module:
            calls = []

            def fake_bridge(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    video_path="video.mp4",
                    frames_path="frames.npy",
                    response_path="response.json",
                    job_dir="job",
                )

            node_module.run_mlx_bridge_job = fake_bridge
            result = node_module.LongCat_Video_SM_MLXGenerate.execute(
                "image",
                "audio",
                "python",
                "weights",
                "q4-merged",
                "dry-run",
                "prompt",
                "negative",
                256,
                432,
                29,
                30,
                7,
                60,
                "jobname",
                True,
            )

        self.assertEqual(result, ("video.mp4", "frames.npy", "response.json", "job"))
        self.assertEqual(calls[0]["runner_python"], "python")
        self.assertEqual(calls[0]["weights_root"], "weights")
        self.assertEqual(calls[0]["variant"], "q4-merged")
        self.assertEqual(calls[0]["mode"], "dry-run")
        self.assertEqual(calls[0]["height"], 256)
        self.assertEqual(calls[0]["width"], 432)
        self.assertEqual(calls[0]["num_frames"], 29)
        self.assertTrue(callable(calls[0]["image_writer"]))
        self.assertTrue(callable(calls[0]["audio_writer"]))

    def test_whisper_and_vocal_model_schemas_use_expected_model_lists(self):
        with loaded_longcat_node_module() as node_module:
            whisper_schema = node_module.LongCat_Video_SM_WhisperModel.define_schema()
            vocal_schema = node_module.LongCat_Video_SM_VocalModel.define_schema()

        self.assertEqual(whisper_schema.display_name, "LongCat Avatar Whisper")
        self.assertEqual(port_by_name(whisper_schema, "audio_encoder").options, ["whisper-large-v3.safetensors"])
        self.assertEqual(vocal_schema.display_name, "LongCat Avatar Vocal Model")
        self.assertEqual(port_by_name(vocal_schema, "audio_encoder_vocal").options, ["none", "vocal.onnx"])

    def test_whisper_model_rejects_non_whisper_encoder_before_loading(self):
        with loaded_longcat_node_module() as node_module:
            def fail_if_called(*args, **kwargs):
                raise AssertionError("invalid Whisper selection should fail before path resolution")

            node_module.folder_paths.get_full_path_or_raise = fail_if_called

            with self.assertRaisesRegex(ValueError, "LongCat Avatar Whisper.*whisper-large-v3"):
                node_module.LongCat_Video_SM_WhisperModel.execute(
                    "wav2vec2-chinese_base_fp16.safetensors",
                )


if __name__ == "__main__":
    unittest.main()
