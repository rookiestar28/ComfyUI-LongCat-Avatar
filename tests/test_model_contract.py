import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from LongCat_Video.model_contract import (
    AVATAR_MAX_GUIDANCE_SCALE,
    AVATAR_MAX_INFERENCE_STEPS,
    AVATAR_MIN_GUIDANCE_SCALE,
    AVATAR_MIN_INFERENCE_STEPS,
    AVATAR_V15,
    OFFICIAL_V15_DISTILL_AUDIO_CFG,
    OFFICIAL_V15_DISTILL_STEPS,
    OFFICIAL_V15_DISTILL_TEXT_CFG,
    detect_dit_format,
    normalize_sampling_parameters,
    resolve_avatar_model_contract,
    safe_display_path,
    validate_state_dict_keys,
    validate_state_dict_result,
)
from LongCat_Video.checkpoint_contract import OFFICIAL_INT8_SHARDED, OFFICIAL_SHARDED
from LongCat_Video.model_loading_contract import resolve_dit_load_plan


RUNTIME_LOADER_PATH = Path("LongCat_Video/run_demo_avatar_single_audio_to_video.py")


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{}")


def _write_index(path, shard_name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            '{"metadata":{"total_size":1},"weight_map":{"layer.weight":"'
            + shard_name
            + '"}}'
        )


def _make_node_layout(root, *, with_int8=True):
    metadata = os.path.join(root, "LongCat_Video", "LongCat-Video")
    _touch(os.path.join(metadata, "tokenizer", "tokenizer_config.json"))
    _touch(os.path.join(metadata, "tokenizer", "spiece.model"))
    _touch(os.path.join(metadata, "scheduler", "scheduler_config.json"))
    _touch(os.path.join(metadata, "vae", "config.json"))
    _touch(os.path.join(metadata, "dit", "config.json"))
    if with_int8:
        _touch(os.path.join(metadata, "base_model_int8", "config.json"))
        _touch(os.path.join(metadata, "base_model_int8", "quantization_config.json"))


class AvatarModelContractTests(unittest.TestCase):
    def test_dit_loader_passes_config_as_positional_argument(self):
        source = RUNTIME_LOADER_PATH.read_text(encoding="utf-8")

        self.assertIn("LongCatVideoAvatarTransformer3DModel.from_config(config)", source)
        self.assertNotIn("LongCatVideoAvatarTransformer3DModel.from_config(**config)", source)

    def test_avatar_v15_requires_distill_lora(self):
        with tempfile.TemporaryDirectory() as root:
            model = os.path.join(root, "model.safetensors")
            vae = os.path.join(root, "vae.safetensors")
            _touch(model)
            _touch(vae)

            with self.assertRaisesRegex(FileNotFoundError, "requires a distill LoRA"):
                resolve_avatar_model_contract(model, vae, None, root)

    def test_resolves_avatar_v15_contract(self):
        with tempfile.TemporaryDirectory() as root:
            _make_node_layout(root)
            model = os.path.join(root, "avatar_int8.safetensors")
            vae = os.path.join(root, "vae.safetensors")
            lora = os.path.join(root, "dmd_lora.safetensors")
            for path in (model, vae, lora):
                _touch(path)

            contract = resolve_avatar_model_contract(
                model,
                vae,
                lora,
                root,
                use_int8=True,
                model_type=AVATAR_V15,
            )

            self.assertEqual(contract.model_type, AVATAR_V15)
            self.assertTrue(contract.use_distill)
            self.assertTrue(contract.use_int8)
            self.assertEqual(contract.effective_num_inference_steps, 8)
            self.assertEqual(contract.effective_text_guidance_scale, 1.0)
            self.assertEqual(contract.effective_audio_guidance_scale, 1.0)
            self.assertEqual(contract.scheduler_source, "embedded_comfy_avatar_v15_scheduler")

    def test_resolves_official_sharded_contract(self):
        with tempfile.TemporaryDirectory() as root:
            _make_node_layout(root)
            checkpoint = os.path.join(root, "LongCat-Video-Avatar-1.5")
            shard = os.path.join(
                checkpoint,
                "base_model",
                "diffusion_pytorch_model-00001-of-00006.safetensors",
            )
            _touch(os.path.join(checkpoint, "base_model", "config.json"))
            _touch(shard)
            _write_index(
                os.path.join(
                    checkpoint,
                    "base_model",
                    "diffusion_pytorch_model.safetensors.index.json",
                ),
                os.path.basename(shard),
            )
            vae = os.path.join(root, "vae.safetensors")
            lora = os.path.join(root, "dmd_lora.safetensors")
            for path in (vae, lora):
                _touch(path)

            contract = resolve_avatar_model_contract(
                None,
                vae,
                lora,
                root,
                checkpoint_source=OFFICIAL_SHARDED,
                official_checkpoint_path=checkpoint,
            )

            self.assertEqual(contract.source_kind, OFFICIAL_SHARDED)
            self.assertEqual(contract.checkpoint_root, checkpoint)
            self.assertEqual(contract.checkpoint_subfolder, "base_model")
            self.assertFalse(contract.use_int8)
            self.assertEqual(contract.checkpoint_shard_paths, (shard,))
            self.assertEqual(resolve_dit_load_plan(contract).loader_kind, "official_sharded")

    def test_dit_load_plan_selects_expected_loader_branches(self):
        with tempfile.TemporaryDirectory() as root:
            _make_node_layout(root)
            model = os.path.join(root, "avatar_int8.safetensors")
            vae = os.path.join(root, "vae.safetensors")
            lora = os.path.join(root, "dmd_lora.safetensors")
            for path in (model, vae, lora):
                _touch(path)

            single_int8 = resolve_avatar_model_contract(
                model,
                vae,
                lora,
                root,
                use_int8=True,
            )
            self.assertEqual(resolve_dit_load_plan(single_int8).loader_kind, "single_file_int8")

            checkpoint = os.path.join(root, "LongCat-Video-Avatar-1.5")
            _touch(os.path.join(checkpoint, "base_model_int8", "config.json"))
            _touch(os.path.join(checkpoint, "base_model_int8", "quantization_config.json"))
            shard = os.path.join(
                checkpoint,
                "base_model_int8",
                "quantized_model-00001-of-00004.safetensors",
            )
            _touch(shard)
            _write_index(
                os.path.join(
                    checkpoint,
                    "base_model_int8",
                    "quantized_model.safetensors.index.json",
                ),
                os.path.basename(shard),
            )
            official_int8 = resolve_avatar_model_contract(
                None,
                vae,
                lora,
                root,
                checkpoint_source=OFFICIAL_INT8_SHARDED,
                official_checkpoint_path=checkpoint,
            )

            plan = resolve_dit_load_plan(official_int8)

            self.assertEqual(plan.loader_kind, "official_sharded_int8")
            self.assertIsNone(plan.single_file)
            self.assertEqual(plan.checkpoint_root, checkpoint)

    def test_int8_rejected_for_unsupported_model_type(self):
        with self.assertRaisesRegex(ValueError, "INT8 inference is only supported"):
            resolve_avatar_model_contract(
                "model.safetensors",
                "vae.safetensors",
                "dmd_lora.safetensors",
                "/tmp/node",
                use_int8=True,
                model_type="avatar-v1.0",
            )

    def test_gguf_rejected_until_supported(self):
        with self.assertRaisesRegex(ValueError, "GGUF DiT loading is not supported"):
            detect_dit_format("/private/models/avatar.gguf")

    def test_int8_metadata_is_required(self):
        with tempfile.TemporaryDirectory() as root:
            _make_node_layout(root, with_int8=False)
            model = os.path.join(root, "avatar_int8.safetensors")
            vae = os.path.join(root, "vae.safetensors")
            lora = os.path.join(root, "dmd_lora.safetensors")
            for path in (model, vae, lora):
                _touch(path)

            with self.assertRaisesRegex(FileNotFoundError, "INT8 model metadata"):
                resolve_avatar_model_contract(model, vae, lora, root, use_int8=True)

    def test_v15_distill_sampling_forces_official_dmd_values(self):
        steps, text_cfg, audio_cfg = normalize_sampling_parameters(
            AVATAR_V15,
            True,
            12,
            3.5,
            5.0,
        )

        self.assertEqual(steps, OFFICIAL_V15_DISTILL_STEPS)
        self.assertEqual(text_cfg, OFFICIAL_V15_DISTILL_TEXT_CFG)
        self.assertEqual(audio_cfg, OFFICIAL_V15_DISTILL_AUDIO_CFG)

    def test_v15_non_distill_sampling_preserves_bounded_user_values(self):
        steps, text_cfg, audio_cfg = normalize_sampling_parameters(
            AVATAR_V15,
            False,
            12,
            3.5,
            5.0,
        )

        self.assertEqual(steps, 12)
        self.assertEqual(text_cfg, 3.5)
        self.assertEqual(audio_cfg, 5.0)

    def test_sampling_parameters_reject_out_of_range_values(self):
        invalid_cases = (
            (AVATAR_MIN_INFERENCE_STEPS - 1, 1.0, 1.0, "steps"),
            (AVATAR_MAX_INFERENCE_STEPS + 1, 1.0, 1.0, "steps"),
            (8, AVATAR_MIN_GUIDANCE_SCALE - 0.1, 1.0, "text_guidance_scale"),
            (8, AVATAR_MAX_GUIDANCE_SCALE + 0.1, 1.0, "text_guidance_scale"),
            (8, 1.0, AVATAR_MIN_GUIDANCE_SCALE - 0.1, "audio_guidance_scale"),
            (8, 1.0, AVATAR_MAX_GUIDANCE_SCALE + 0.1, "audio_guidance_scale"),
        )

        for steps, text_cfg, audio_cfg, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    normalize_sampling_parameters(AVATAR_V15, True, steps, text_cfg, audio_cfg)

    def test_state_dict_result_rejects_mismatch(self):
        result = SimpleNamespace(
            missing_keys=["encoder.block.weight"],
            unexpected_keys=["unused.weight"],
        )

        with self.assertRaisesRegex(ValueError, "state dict mismatch"):
            validate_state_dict_result("DiT", result)

    def test_state_dict_policy_allows_explicit_prefixes(self):
        validate_state_dict_keys(
            "VAE",
            ["optional_adapter.weight"],
            ["debug_head.weight"],
            allow_missing_prefixes=("optional_adapter.",),
            allow_unexpected_prefixes=("debug_head.",),
        )

    def test_safe_display_path_uses_basename(self):
        self.assertEqual(
            safe_display_path("/private/user/models/avatar.safetensors"),
            "avatar.safetensors",
        )


if __name__ == "__main__":
    unittest.main()
