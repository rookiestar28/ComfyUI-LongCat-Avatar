import json
import os
import tempfile
import unittest

from LongCat_Video.checkpoint_contract import (
    OFFICIAL_INT8_SHARDED,
    OFFICIAL_SHARDED,
    SINGLE_FILE_SAFETENSORS,
    describe_checkpoint_source_role,
    inspect_checkpoint_source,
    is_official_checkpoint_source,
    is_single_file_checkpoint_source,
    validate_checkpoint_source,
)


def _touch(path, content=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _write_index(path, weight_map):
    _touch(
        path,
        json.dumps(
            {
                "metadata": {"total_size": 1234},
                "weight_map": weight_map,
            }
        ),
    )


def _make_official_checkpoint(root, *, int8=False):
    subfolder = "base_model_int8" if int8 else "base_model"
    index_name = (
        "quantized_model.safetensors.index.json"
        if int8
        else "diffusion_pytorch_model.safetensors.index.json"
    )
    shard_name = (
        "quantized_model-00001-of-00004.safetensors"
        if int8
        else "diffusion_pytorch_model-00001-of-00006.safetensors"
    )
    _touch(os.path.join(root, subfolder, "config.json"), "{}")
    if int8:
        _touch(os.path.join(root, subfolder, "quantization_config.json"), "{}")
    _touch(os.path.join(root, subfolder, shard_name), "fake")
    _write_index(
        os.path.join(root, subfolder, index_name),
        {"layer.weight": shard_name},
    )


class ShardedCheckpointContractTests(unittest.TestCase):
    def test_source_kind_roles_distinguish_official_from_local_converted_inputs(self):
        self.assertFalse(is_official_checkpoint_source(SINGLE_FILE_SAFETENSORS))
        self.assertTrue(is_single_file_checkpoint_source(SINGLE_FILE_SAFETENSORS))
        self.assertEqual(
            describe_checkpoint_source_role(SINGLE_FILE_SAFETENSORS),
            "local_or_community_converted_single_file",
        )

        for source_kind in (OFFICIAL_SHARDED, OFFICIAL_INT8_SHARDED):
            with self.subTest(source_kind=source_kind):
                self.assertTrue(is_official_checkpoint_source(source_kind))
                self.assertFalse(is_single_file_checkpoint_source(source_kind))
                self.assertEqual(describe_checkpoint_source_role(source_kind), "official_sharded")

    def test_single_file_source_is_validated(self):
        with tempfile.TemporaryDirectory() as root:
            model = os.path.join(root, "avatar.safetensors")
            _touch(model, "fake")

            spec = validate_checkpoint_source(SINGLE_FILE_SAFETENSORS, model)

            self.assertEqual(spec.source_kind, SINGLE_FILE_SAFETENSORS)
            self.assertEqual(spec.model_path, model)
            self.assertFalse(spec.use_int8)

    def test_official_sharded_source_validates_base_model(self):
        with tempfile.TemporaryDirectory() as root:
            _make_official_checkpoint(root, int8=False)

            spec = validate_checkpoint_source(OFFICIAL_SHARDED, root)

            self.assertEqual(spec.source_kind, OFFICIAL_SHARDED)
            self.assertEqual(spec.subfolder, "base_model")
            self.assertEqual(
                spec.index_name,
                "diffusion_pytorch_model.safetensors.index.json",
            )
            self.assertEqual(
                spec.shard_names,
                ("diffusion_pytorch_model-00001-of-00006.safetensors",),
            )
            self.assertFalse(spec.use_int8)

    def test_official_int8_source_validates_quantized_model(self):
        with tempfile.TemporaryDirectory() as root:
            _make_official_checkpoint(root, int8=True)

            spec = validate_checkpoint_source(OFFICIAL_INT8_SHARDED, root)

            self.assertEqual(spec.source_kind, OFFICIAL_INT8_SHARDED)
            self.assertEqual(spec.subfolder, "base_model_int8")
            self.assertTrue(spec.use_int8)
            self.assertIn("base_model_int8/quantization_config.json", spec.required_files)

    def test_missing_assets_are_reported_without_loading(self):
        with tempfile.TemporaryDirectory() as root:
            _touch(os.path.join(root, "base_model", "config.json"), "{}")

            inspection = inspect_checkpoint_source(OFFICIAL_SHARDED, root)

            self.assertFalse(inspection.is_complete)
            self.assertIsNone(inspection.spec)
            self.assertIn(
                "base_model/diffusion_pytorch_model.safetensors.index.json",
                inspection.missing_files,
            )

    def test_malformed_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            _touch(os.path.join(root, "base_model", "config.json"), "{}")
            _touch(
                os.path.join(
                    root,
                    "base_model",
                    "diffusion_pytorch_model.safetensors.index.json",
                ),
                "{not-json",
            )

            with self.assertRaisesRegex(ValueError, "Malformed sharded checkpoint index"):
                validate_checkpoint_source(OFFICIAL_SHARDED, root)

    def test_unsafe_shard_references_are_rejected(self):
        unsafe_values = (
            "../escape.safetensors",
            "/absolute/escape.safetensors",
            "weights.bin",
            "",
        )
        for shard_name in unsafe_values:
            with self.subTest(shard_name=shard_name):
                with tempfile.TemporaryDirectory() as root:
                    _touch(os.path.join(root, "base_model", "config.json"), "{}")
                    _write_index(
                        os.path.join(
                            root,
                            "base_model",
                            "diffusion_pytorch_model.safetensors.index.json",
                        ),
                        {"layer.weight": shard_name},
                    )

                    with self.assertRaisesRegex(ValueError, "Unsafe shard reference|Unsupported shard file"):
                        validate_checkpoint_source(OFFICIAL_SHARDED, root)

    def test_missing_referenced_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            _touch(os.path.join(root, "base_model", "config.json"), "{}")
            _write_index(
                os.path.join(
                    root,
                    "base_model",
                    "diffusion_pytorch_model.safetensors.index.json",
                ),
                {"layer.weight": "missing.safetensors"},
            )

            with self.assertRaisesRegex(FileNotFoundError, "Missing checkpoint shard"):
                validate_checkpoint_source(OFFICIAL_SHARDED, root)


if __name__ == "__main__":
    unittest.main()
