import os
import tempfile
import unittest

from LongCat_Video.checkpoint_contract import (
    OFFICIAL_INT8_SHARDED,
    OFFICIAL_SHARDED,
    build_download_manifest,
    build_text_encoder_download_manifest,
    download_missing_checkpoint_assets,
)


class ModelDownloadContractTests(unittest.TestCase):
    def test_manifest_is_fixed_to_official_avatar_repo(self):
        with tempfile.TemporaryDirectory() as models_dir:
            manifest = build_download_manifest(OFFICIAL_SHARDED, models_dir)

            self.assertEqual(manifest.repo_id, "meituan-longcat/LongCat-Video-Avatar-1.5")
            self.assertIn("base_model/diffusion_pytorch_model-*.safetensors", manifest.allow_patterns)
            self.assertTrue(manifest.local_dir.startswith(os.path.abspath(models_dir)))
            self.assertNotIn("http", " ".join(manifest.allow_patterns))

    def test_int8_manifest_includes_quantization_assets(self):
        with tempfile.TemporaryDirectory() as models_dir:
            manifest = build_download_manifest(OFFICIAL_INT8_SHARDED, models_dir)

            self.assertIn("base_model_int8/quantization_config.json", manifest.allow_patterns)
            self.assertIn("base_model_int8/quantized_model-*.safetensors", manifest.allow_patterns)

    def test_downloader_uses_snapshot_download_with_allow_patterns(self):
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return kwargs["local_dir"]

        with tempfile.TemporaryDirectory() as models_dir:
            manifest = build_download_manifest(OFFICIAL_SHARDED, models_dir)
            result = download_missing_checkpoint_assets(
                manifest,
                snapshot_download=fake_snapshot_download,
            )

        self.assertEqual(result.local_dir, manifest.local_dir)
        self.assertEqual(calls[0]["repo_id"], manifest.repo_id)
        self.assertEqual(calls[0]["allow_patterns"], manifest.allow_patterns)
        self.assertEqual(calls[0]["local_dir"], manifest.local_dir)
        self.assertFalse(calls[0]["local_files_only"])

    def test_downloader_rejects_target_outside_model_root(self):
        with tempfile.TemporaryDirectory() as models_dir:
            with self.assertRaisesRegex(ValueError, "download target"):
                build_download_manifest(OFFICIAL_SHARDED, models_dir, model_dir_name="../escape")

    def test_unknown_source_kind_has_no_manifest(self):
        with tempfile.TemporaryDirectory() as models_dir:
            with self.assertRaisesRegex(ValueError, "No download manifest"):
                build_download_manifest("single_file_safetensors", models_dir)

    def test_text_encoder_manifest_is_fixed_to_official_base_repo(self):
        with tempfile.TemporaryDirectory() as models_dir:
            manifest = build_text_encoder_download_manifest(models_dir)

            self.assertEqual(manifest.repo_id, "meituan-longcat/LongCat-Video")
            self.assertEqual(manifest.model_dir_name, "LongCat-Video")
            self.assertIn("tokenizer/*", manifest.allow_patterns)
            self.assertIn("text_encoder/*", manifest.allow_patterns)
            self.assertTrue(manifest.local_dir.startswith(os.path.abspath(models_dir)))


if __name__ == "__main__":
    unittest.main()
