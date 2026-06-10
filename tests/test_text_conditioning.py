import os
import tempfile
import unittest
from unittest.mock import patch

from LongCat_Video.text_conditioning import (
    DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT,
    EXPECTED_TEXT_HIDDEN_SIZE,
    MAX_TEXT_SEQUENCE_LENGTH,
    TEXT_CONDITIONING_SOURCE_CLIP,
    TEXT_CONDITIONING_SOURCE_OFFICIAL,
    normalize_text_encoder_offload_device,
    extract_scheduled_text_embedding,
    resolve_or_download_official_text_encoder_layout,
    resolve_official_text_encoder_layout,
    validate_final_text_embedding,
    validate_scheduled_text_embedding,
    validate_text_conditioning_payload,
)


class FakeTensor:
    def __init__(self, shape, *, finite=True, dtype="bfloat16", device="cpu"):
        self.shape = tuple(shape)
        self.finite = finite
        self.dtype = dtype
        self.device = device

    def isfinite(self):
        return FakeFinite(self.finite)


class FakeFinite:
    def __init__(self, finite):
        self.finite = finite

    def all(self):
        return self

    def item(self):
        return self.finite


def _valid_rank3(seq_len=MAX_TEXT_SEQUENCE_LENGTH):
    return FakeTensor((1, seq_len, EXPECTED_TEXT_HIDDEN_SIZE))


def _valid_rank4(seq_len=MAX_TEXT_SEQUENCE_LENGTH):
    return FakeTensor((1, 1, seq_len, EXPECTED_TEXT_HIDDEN_SIZE))


class TextConditioningTests(unittest.TestCase):
    def test_extracts_valid_scheduled_embedding(self):
        embedding = _valid_rank3()
        extracted = extract_scheduled_text_embedding([[embedding]], "prompt_embeds")

        self.assertIs(extracted, embedding)

    def test_rejects_malformed_scheduled_output(self):
        with self.assertRaisesRegex(ValueError, "malformed"):
            extract_scheduled_text_embedding([[]], "prompt_embeds")

    def test_rejects_rank3_hidden_mismatch(self):
        with self.assertRaisesRegex(ValueError, "hidden size"):
            validate_scheduled_text_embedding(FakeTensor((1, 512, 2048)), "prompt_embeds")

    def test_rejects_rank4_hidden_mismatch(self):
        with self.assertRaisesRegex(ValueError, "hidden size"):
            validate_final_text_embedding(FakeTensor((1, 1, 512, 2048)), "prompt_embeds")

    def test_rejects_sequence_too_long(self):
        with self.assertRaisesRegex(ValueError, "sequence length"):
            validate_final_text_embedding(FakeTensor((1, 1, 513, EXPECTED_TEXT_HIDDEN_SIZE)), "prompt_embeds")

    def test_rejects_non_finite_values_when_available(self):
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            validate_final_text_embedding(FakeTensor((1, 1, 512, EXPECTED_TEXT_HIDDEN_SIZE), finite=False), "prompt_embeds")

    def test_validates_complete_conditioning_payload(self):
        validate_text_conditioning_payload(
            {
                "prompt_embeds": _valid_rank4(),
                "negative_prompt_embeds": _valid_rank4(),
                "text": ["prompt", "negative"],
                "conditioning_source": TEXT_CONDITIONING_SOURCE_CLIP,
            },
            require_negative=True,
        )

    def test_missing_negative_prompt_fails_before_generation(self):
        with self.assertRaisesRegex(KeyError, "negative_prompt_embeds"):
            validate_text_conditioning_payload(
                {
                    "prompt_embeds": _valid_rank4(),
                    "text": ["prompt", "negative"],
                },
                require_negative=True,
            )

    def test_negative_prompt_shape_must_match(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            validate_text_conditioning_payload(
                {
                    "prompt_embeds": _valid_rank4(512),
                    "negative_prompt_embeds": _valid_rank4(256),
                    "text": ["prompt", "negative"],
                },
                require_negative=True,
            )

    def test_errors_do_not_include_prompt_text(self):
        prompt_text = "private prompt contents should not appear"
        with self.assertRaises(ValueError) as caught:
            validate_text_conditioning_payload(
                {
                    "prompt_embeds": FakeTensor((1, 1, 512, 2048)),
                    "negative_prompt_embeds": _valid_rank4(),
                    "text": [prompt_text, "negative"],
                },
                require_negative=True,
            )

        self.assertNotIn(prompt_text, str(caught.exception))

    def test_resolves_official_shared_longcat_text_encoder_layout(self):
        with tempfile.TemporaryDirectory() as root:
            model_root = os.path.join(root, DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT)
            os.makedirs(os.path.join(model_root, "tokenizer"))
            os.makedirs(os.path.join(model_root, "text_encoder"))
            open(os.path.join(model_root, "tokenizer", "tokenizer_config.json"), "w", encoding="utf-8").close()
            open(os.path.join(model_root, "text_encoder", "config.json"), "w", encoding="utf-8").close()
            open(os.path.join(model_root, "text_encoder", "model.safetensors"), "w", encoding="utf-8").close()

            layout = resolve_official_text_encoder_layout(DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT, root)

        self.assertEqual(layout.root_name, DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT)
        self.assertTrue(layout.tokenizer_dir.endswith(os.path.join(DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT, "tokenizer")))
        self.assertTrue(layout.text_encoder_dir.endswith(os.path.join(DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT, "text_encoder")))

    def test_resolves_selection_inside_text_encoder_back_to_shared_root(self):
        with tempfile.TemporaryDirectory() as root:
            model_root = os.path.join(root, DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT)
            os.makedirs(os.path.join(model_root, "tokenizer"))
            os.makedirs(os.path.join(model_root, "text_encoder"))
            open(os.path.join(model_root, "tokenizer", "tokenizer_config.json"), "w", encoding="utf-8").close()
            open(os.path.join(model_root, "text_encoder", "config.json"), "w", encoding="utf-8").close()
            open(
                os.path.join(model_root, "text_encoder", "model.safetensors.index.json"),
                "w",
                encoding="utf-8",
            ).close()

            layout = resolve_official_text_encoder_layout("LongCat-Video/text_encoder/config.json", root)

        self.assertEqual(layout.root_name, DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT)

    def test_official_text_encoder_layout_rejects_missing_shared_assets(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(FileNotFoundError, "shared base LongCat-Video text encoder"):
                resolve_official_text_encoder_layout(DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT, root)

    def test_official_text_encoder_layout_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "inside ComfyUI/models/longcat"):
                resolve_official_text_encoder_layout("../LongCat-Video", root)

    def test_payload_accepts_official_conditioning_source_metadata(self):
        validate_text_conditioning_payload(
            {
                "prompt_embeds": _valid_rank4(),
                "negative_prompt_embeds": _valid_rank4(),
                "text": ["prompt", "negative"],
                "conditioning_source": TEXT_CONDITIONING_SOURCE_OFFICIAL,
                "text_encoder_root": DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT,
            },
            require_negative=True,
        )

    def test_normalizes_text_encoder_offload_device(self):
        self.assertEqual(normalize_text_encoder_offload_device("cpu"), "cpu")
        self.assertEqual(normalize_text_encoder_offload_device("CUDA"), "cuda")
        with self.assertRaisesRegex(ValueError, "offload_device"):
            normalize_text_encoder_offload_device("mps")

    def test_official_text_encoder_layout_auto_downloads_default_root(self):
        with tempfile.TemporaryDirectory() as root:
            def fake_download(manifest):
                model_root = os.path.join(root, DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT)
                os.makedirs(os.path.join(model_root, "tokenizer"))
                os.makedirs(os.path.join(model_root, "text_encoder"))
                open(os.path.join(model_root, "tokenizer", "tokenizer_config.json"), "w", encoding="utf-8").close()
                open(os.path.join(model_root, "text_encoder", "config.json"), "w", encoding="utf-8").close()
                open(os.path.join(model_root, "text_encoder", "model.safetensors"), "w", encoding="utf-8").close()
                return object()

            with patch("LongCat_Video.text_conditioning.download_missing_checkpoint_assets", side_effect=fake_download) as mocked:
                layout = resolve_or_download_official_text_encoder_layout(
                    DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT,
                    root,
                    auto_download_missing_text_encoder=True,
                )

        self.assertEqual(layout.root_name, DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT)
        self.assertEqual(mocked.call_count, 1)

    def test_official_text_encoder_auto_download_rejects_custom_root(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(FileNotFoundError, "only supported for LongCat-Video"):
                resolve_or_download_official_text_encoder_layout(
                    "Custom-LongCat",
                    root,
                    auto_download_missing_text_encoder=True,
                )


if __name__ == "__main__":
    unittest.main()
