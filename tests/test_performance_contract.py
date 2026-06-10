import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from LongCat_Video.performance_contract import (
    MAX_STREAMING_PREFETCH_BLOCKS,
    VAE_OFFLOAD_DEVICES,
    apply_runtime_plan,
    build_runtime_plan,
    cleanup_runtime_plan,
    normalize_block_num,
    normalize_device_name,
    normalize_offload_device,
    validate_precision_runtime_request,
    require_cuda_device,
)


ATTENTION_SOURCE_PATHS = (
    Path("LongCat_Video/longcat_video/modules/avatar/attention.py"),
    Path("LongCat_Video/longcat_video/modules/attention.py"),
)
AVATAR_GENERATION_SOURCE_PATHS = (
    Path("LongCat_Video/run_demo_avatar_single_audio_to_video.py"),
    Path("LongCat_Video/run_demo_avatar_multi_audio_to_video.py"),
)


class FakeModel:
    def __init__(self):
        self.calls = []
        self.streaming_prefetch_count = "unset"
        self.vae_offload_device = "unset"
        self.dit = SimpleNamespace(lora_runtime_offload="unset")

    def vae_to(self, device):
        self.calls.append(("vae_to", device))

    def to(self, device):
        self.calls.append(("to", device))


class PerformanceContractTests(unittest.TestCase):
    def test_normalizes_device_like_objects(self):
        self.assertEqual(normalize_device_name("cuda:0"), "cuda:0")
        self.assertEqual(normalize_device_name(SimpleNamespace(type="cuda", index=1)), "cuda:1")
        self.assertEqual(normalize_device_name(SimpleNamespace(type="cuda", index=None)), "cuda")

    def test_rejects_non_cuda_devices_early(self):
        for value in ("cpu", "mps", SimpleNamespace(type="mps", index=None)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
                    require_cuda_device(value)

    def test_normalizes_block_bounds(self):
        self.assertEqual(normalize_block_num("0"), 0)
        self.assertEqual(normalize_block_num(MAX_STREAMING_PREFETCH_BLOCKS), 64)
        with self.assertRaisesRegex(ValueError, "between 0 and"):
            normalize_block_num(-1)
        with self.assertRaisesRegex(ValueError, "between 0 and"):
            normalize_block_num(MAX_STREAMING_PREFETCH_BLOCKS + 1)
        with self.assertRaisesRegex(ValueError, "integer"):
            normalize_block_num("bad")

    def test_normalizes_vae_offload_device(self):
        self.assertEqual(VAE_OFFLOAD_DEVICES, ("cpu", "cuda"))
        self.assertEqual(normalize_offload_device("cpu"), "cpu")
        self.assertEqual(normalize_offload_device("CUDA"), "cuda")
        with self.assertRaisesRegex(ValueError, "offload_device"):
            normalize_offload_device("mps")

    def test_block_zero_uses_eager_full_load_and_cleanup(self):
        plan = build_runtime_plan("cuda:0", 0)
        model = FakeModel()
        empty_cache_calls = []

        apply_runtime_plan(model, plan)

        self.assertIsNone(plan.streaming_prefetch_count)
        self.assertTrue(plan.move_dit_to_device)
        self.assertTrue(plan.offload_dit_after_generate)
        self.assertEqual(plan.vae_offload_device, "cpu")
        self.assertEqual(model.streaming_prefetch_count, None)
        self.assertEqual(model.vae_offload_device, "cpu")
        self.assertFalse(model.dit.lora_runtime_offload)

        cleanup_runtime_plan(model, plan, empty_cache=lambda: empty_cache_calls.append("empty"))

        self.assertEqual(model.calls, [("vae_to", "cuda:0"), ("to", "cuda:0"), ("to", "cpu")])
        self.assertTrue(model.dit.lora_runtime_offload)
        self.assertEqual(empty_cache_calls, ["empty"])

    def test_positive_block_uses_streaming_prefetch_without_dit_offload(self):
        plan = build_runtime_plan(SimpleNamespace(type="cuda", index=0), 3, "cuda")
        model = FakeModel()
        empty_cache_calls = []

        apply_runtime_plan(model, plan)
        cleanup_runtime_plan(model, plan, empty_cache=lambda: empty_cache_calls.append("empty"))

        self.assertEqual(plan.streaming_prefetch_count, 3)
        self.assertFalse(plan.move_dit_to_device)
        self.assertFalse(plan.offload_dit_after_generate)
        self.assertEqual(plan.vae_offload_device, "cuda")
        self.assertEqual(model.streaming_prefetch_count, 3)
        self.assertEqual(model.vae_offload_device, "cuda")
        self.assertTrue(model.dit.lora_runtime_offload)
        self.assertEqual(model.calls, [("vae_to", "cuda:0")])
        self.assertEqual(empty_cache_calls, [])

    def test_avatar_lora_forward_keeps_full_load_resident_guard(self):
        source = Path("LongCat_Video/longcat_video/modules/avatar/longcat_video_dit_avatar.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("lora_runtime_offload", source)
        self.assertIn("block_num=0 is the full-load path", source)
        self.assertIn("if offload_after_forward:", source)
        self.assertNotIn("# 3. 推理后：立即将 LoRA 权重卸载回 CPU", source)

    def test_pipeline_vae_offload_uses_runtime_device(self):
        source = Path("LongCat_Video/longcat_video/pipeline_longcat_video_avatar.py").read_text(encoding="utf-8")

        self.assertIn('self.vae_offload_device = "cpu"', source)
        self.assertIn("offload_device=cuda intentionally keeps VAE resident", source)
        self.assertIn('getattr(self, "vae_offload_device", "cpu")', source)
        self.assertNotIn('self.vae = self.vae.to("cpu")', source)

    def test_pipeline_uses_nullcontext_for_non_streaming_model_context(self):
        source = Path("LongCat_Video/longcat_video/pipeline_longcat_video_avatar.py").read_text(encoding="utf-8")

        self.assertIn("return nullcontext(self.dit)", source)
        self.assertNotIn("if self.streaming_prefetch_count is not None else self.dit", source)
        self.assertNotIn("model_context = self.dit", source)
        self.assertEqual(source.count("model_context = self._model_ctx(self.streaming_prefetch_count)"), 4)

    def test_streaming_model_teardown_removes_forward_hooks(self):
        source = Path("LongCat_Video/layer_streaming.py").read_text(encoding="utf-8")

        self.assertIn("self._hook_handles", source)
        self.assertIn("self._hook_handles.extend((pre_hook, post_hook))", source)
        self.assertIn("handle.remove()", source)
        self.assertIn("self._hook_handles.clear()", source)
        self.assertIn("every segment stacks hooks and slows denoising", source)
        self.assertEqual(source.count("wrapped.teardown()"), 2)
        self.assertNotIn('wrapped.to("cpu")', source)

    def test_sampler_logs_effective_runtime_plan(self):
        source = Path("LongCat_Video_node.py").read_text(encoding="utf-8")

        self.assertIn("LongCat runtime plan", source)
        self.assertIn("streaming_prefetch_count", source)
        self.assertIn("offload_dit_after_generate", source)

    def test_debug_profiler_is_opt_in_and_memory_only(self):
        source = Path("LongCat_Video/debug_profile.py").read_text(encoding="utf-8")

        self.assertIn("class LongCatDebugProfiler", source)
        self.assertIn("if not self.enabled", source)
        self.assertIn("torch.cuda.synchronize", source)
        self.assertIn("cuda_alloc_gb", source)
        self.assertIn("elapsed_s", source)
        self.assertNotIn("prompt", source)
        self.assertNotIn("audio_path", source)
        self.assertNotIn("checkpoint", source)

    def test_sampler_threads_debug_profile_to_generation_paths(self):
        source = Path("LongCat_Video_node.py").read_text(encoding="utf-8")

        self.assertIn('io.Boolean.Input("debug_mode", default=False)', source)
        self.assertIn("LongCatDebugProfiler(bool(debug_mode)", source)
        self.assertIn('debug_profile.phase("apply_runtime_plan")', source)
        self.assertIn('debug_profile.phase("generate"', source)
        self.assertIn("debug_profile=debug_profile.child(\"single\")", source)
        self.assertIn("debug_profile=debug_profile.child(\"multi\")", source)

    def test_avatar_pipeline_debug_profile_covers_hot_phases(self):
        source = Path("LongCat_Video/longcat_video/pipeline_longcat_video_avatar.py").read_text(encoding="utf-8")

        self.assertEqual(source.count("debug_profile=None"), 3)
        self.assertEqual(source.count("ensure_debug_profiler(debug_profile)"), 3)
        self.assertIn("kv_cache_prepare", source)
        self.assertGreaterEqual(source.count("denoising_loop"), 5)
        self.assertGreaterEqual(source.count("vae_decode"), 3)
        self.assertGreaterEqual(source.count("postprocess_video"), 3)

    def test_avatar_continuation_keeps_kv_cache_on_gpu(self):
        for path in AVATAR_GENERATION_SOURCE_PATHS:
            with self.subTest(path=str(path)):
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
                generate_avc_calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "generate_avc"
                ]

                self.assertTrue(generate_avc_calls, f"{path} must call generate_avc")
                for call in generate_avc_calls:
                    kwargs = {keyword.arg: keyword.value for keyword in call.keywords}
                    self.assertIn("use_kv_cache", kwargs)
                    self.assertIn("offload_kv_cache", kwargs)
                    self.assertIs(kwargs["use_kv_cache"].value, True)
                    self.assertIs(kwargs["offload_kv_cache"].value, False)
                self.assertIn("official Avatar continuation keeps KV cache on GPU", source)

    def test_runtime_plan_does_not_include_sampler_input_mutations(self):
        plan = build_runtime_plan("cuda:0", 1)

        self.assertFalse(hasattr(plan, "seed"))
        self.assertFalse(hasattr(plan, "stage_1"))
        self.assertFalse(hasattr(plan, "resolution"))
        self.assertFalse(hasattr(plan, "text_guidance_scale"))
        self.assertFalse(hasattr(plan, "audio_guidance_scale"))

    def test_precision_runtime_request_accepts_official_default(self):
        plan = validate_precision_runtime_request()

        self.assertEqual(plan.base_precision, "bf16")
        self.assertEqual(plan.fp8_mode, "disabled")
        self.assertEqual(plan.quantization_source, "none")

    def test_precision_runtime_request_tracks_official_int8_as_weight_mode(self):
        plan = validate_precision_runtime_request(checkpoint_source="official_int8_sharded")

        self.assertEqual(plan.base_precision, "bf16")
        self.assertEqual(plan.quantization_source, "official_int8_sharded")

    def test_precision_runtime_request_rejects_fp16_until_supported(self):
        with self.assertRaisesRegex(NotImplementedError, "FP16"):
            validate_precision_runtime_request(base_precision="fp16")

    def test_precision_runtime_request_rejects_fp8_until_supported(self):
        with self.assertRaisesRegex(NotImplementedError, "FP8"):
            validate_precision_runtime_request(fp8_mode="fp8_e4m3fn")

    def test_precision_runtime_request_rejects_fp8_on_low_cuda_capability(self):
        with self.assertRaisesRegex(ValueError, "CUDA compute capability"):
            validate_precision_runtime_request(fp8_mode="fp8_e4m3fn_fast", cuda_capability=(8, 6))

    def test_precision_runtime_request_rejects_gguf_until_supported(self):
        with self.assertRaisesRegex(ValueError, "GGUF"):
            validate_precision_runtime_request(gguf_model="avatar.gguf")

    def test_attention_modules_guard_flash_attention_with_sdpa_fallback(self):
        helper_source = Path("LongCat_Video/longcat_video/modules/attention_ops.py").read_text(encoding="utf-8")
        self.assertIn("scaled_dot_product_attention", helper_source)

        for path in ATTENTION_SOURCE_PATHS:
            with self.subTest(path=str(path)):
                source = path.read_text(encoding="utf-8")

                self.assertIn("_callable_or_none", source)
                self.assertIn("_sdpa_attention", source)
                self.assertIn("callable(flash_attn_func)", source)
                self.assertNotIn("x = flash_attn_func(\n                q,", source)

    def test_attention_modules_include_lazy_sageattention_fallback(self):
        helper_source = Path("LongCat_Video/longcat_video/modules/attention_ops.py").read_text(encoding="utf-8")
        self.assertIn('callable_or_none("sageattention", "sageattn")', helper_source)
        self.assertIn('callable_or_none("sageattn3", "sageattn3_blackwell")', helper_source)
        self.assertIn("warn_attention_fallback", helper_source)
        self.assertNotIn("from sageattention import", helper_source)
        self.assertNotIn("from sageattn3 import", helper_source)

        for path in ATTENTION_SOURCE_PATHS:
            with self.subTest(path=str(path)):
                source = path.read_text(encoding="utf-8")

                self.assertIn("enable_sageattn", source)
                self.assertIn("enable_sageattn3", source)
                self.assertIn("sage_attention", source)
                self.assertIn("_sdpa_attention", source)
                self.assertIn("warn_attention_fallback", source)


if __name__ == "__main__":
    unittest.main()
