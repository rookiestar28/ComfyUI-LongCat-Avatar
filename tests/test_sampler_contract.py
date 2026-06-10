import unittest

from LongCat_Video.audio_contract import build_avatar_audio_payload
from LongCat_Video.sampler_contract import (
    MAX_SEED,
    MULTI_MODE,
    SINGLE_MODE,
    build_audio_window_payload,
    build_audio_window_spec,
    build_latent_bookkeeping_spec,
    classify_audio_payload_window_state,
    expected_output_frames,
    normalize_seed,
    resolve_resolution_dimensions,
    segment_audio_start,
    select_sampler_mode,
    trim_output_tensor_to_target_frames,
    validate_output_tensor_contract,
    validate_sampler_inputs,
)


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeSliceableTensor(FakeTensor):
    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self.shape[0])
            frame_count = max(0, (stop - start + (step - 1)) // step)
            return FakeSliceableTensor((frame_count, *self.shape[1:]))
        raise TypeError("FakeSliceableTensor only supports frame slicing.")


class FakeIndexableAudio:
    shape = (200, 5, 1280)

    def __getitem__(self, item):
        return ("slice", item)


class SamplerContractTests(unittest.TestCase):
    def test_selects_single_mode_without_left_audio(self):
        self.assertEqual(select_sampler_mode({"left_full_audio_emb": None}), SINGLE_MODE)

    def test_selects_multi_mode_with_left_audio(self):
        self.assertEqual(select_sampler_mode({"left_full_audio_emb": object()}), MULTI_MODE)

    def test_single_mode_accepts_at2v_and_ai2v(self):
        base = {"left_full_audio_emb": None}
        self.assertEqual(
            validate_sampler_inputs(base, stage_1="at2v", resolution="480p", seed=1, ref_img_index=10, mask_frame_range=3),
            SINGLE_MODE,
        )
        self.assertEqual(
            validate_sampler_inputs(base, stage_1="ai2v", resolution="480p", seed=1, ref_img_index=10, mask_frame_range=3),
            SINGLE_MODE,
        )

    def test_multi_mode_rejects_at2v(self):
        with self.assertRaisesRegex(ValueError, "image-conditioned"):
            validate_sampler_inputs(
                {"left_full_audio_emb": object()},
                stage_1="at2v",
                resolution="480p",
                seed=1,
                ref_img_index=10,
                mask_frame_range=3,
            )

    def test_resolution_maps_to_official_dimensions(self):
        self.assertEqual(resolve_resolution_dimensions("480p"), (480, 832))
        self.assertEqual(resolve_resolution_dimensions("720p"), (768, 1280))

    def test_resolution_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "Unsupported resolution"):
            resolve_resolution_dimensions("1080p")

    def test_seed_bounds_are_deterministic(self):
        self.assertEqual(normalize_seed("42"), 42)
        self.assertEqual(normalize_seed(MAX_SEED), MAX_SEED)
        with self.assertRaisesRegex(ValueError, "seed must be between"):
            normalize_seed(MAX_SEED + 1)

    def test_continuation_audio_start_matches_official_window(self):
        self.assertEqual(segment_audio_start(0, 1), 0)
        self.assertEqual(segment_audio_start(1, 1), 80)
        self.assertEqual(segment_audio_start(2, 1), 160)

    def test_expected_output_frames_match_overlap_math(self):
        self.assertEqual(expected_output_frames(1), 93)
        self.assertEqual(expected_output_frames(3), 253)
        self.assertEqual(expected_output_frames(4), 333)

    def test_trims_segment_envelope_to_audio_target_frames(self):
        output = FakeSliceableTensor((expected_output_frames(4), 8, 8, 3))

        trimmed = trim_output_tensor_to_target_frames(output, 250)

        self.assertEqual(tuple(trimmed.shape), (250, 8, 8, 3))

    def test_trim_rejects_target_beyond_generated_frames(self):
        output = FakeSliceableTensor((expected_output_frames(4), 8, 8, 3))

        with self.assertRaisesRegex(ValueError, "exceeds generated frame count"):
            trim_output_tensor_to_target_frames(output, 334)

    def test_audio_window_spec_first_segment_clamps_center_context(self):
        spec = build_audio_window_spec(
            frames_processed=0,
            num_frames=93,
            overlap=13,
            audio_stride=1,
            full_audio_frames=200,
        )

        self.assertEqual(spec.audio_start_idx, 0)
        self.assertEqual(spec.audio_end_idx, 93)
        self.assertEqual(spec.center_indices[0], (0, 0, 0, 1, 2))

    def test_audio_window_spec_later_segment_uses_official_overlap_math(self):
        spec = build_audio_window_spec(
            frames_processed=93,
            num_frames=93,
            overlap=13,
            audio_stride=1,
            full_audio_frames=200,
        )

        self.assertEqual(spec.audio_start_idx, segment_audio_start(1, 1))
        self.assertEqual(spec.audio_end_idx, 173)
        self.assertEqual(spec.center_indices[0], (78, 79, 80, 81, 82))

    def test_audio_window_payload_carries_slices_metadata_and_optional_samples(self):
        audio = FakeIndexableAudio()
        payload = build_avatar_audio_payload(full_audio_emb=audio, num_segments=2)

        window_payload = build_audio_window_payload(
            payload,
            frames_processed=93,
            num_frames=93,
            overlap=13,
            samples={"samples": "latent"},
        )

        self.assertEqual(window_payload["window"]["audio_start_idx"], 80)
        self.assertEqual(window_payload["audio_emb_slice"][0][0], "slice")
        self.assertEqual(window_payload["samples_slice"], {"samples": "latent"})
        self.assertEqual(classify_audio_payload_window_state(window_payload), "sliced")

    def test_audio_payload_window_state_detects_full_payload(self):
        payload = build_avatar_audio_payload(full_audio_emb=FakeIndexableAudio(), num_segments=1)

        self.assertEqual(classify_audio_payload_window_state(payload), "full")

    def test_latent_bookkeeping_offsets_clean_indices_after_reference_latents(self):
        spec = build_latent_bookkeeping_spec(
            clean_latent_indices=(0, 2),
            ref_latent_count=1,
            generated_latent_count=5,
        )

        self.assertEqual(spec.ref_latent_indices, (0,))
        self.assertEqual(spec.adjusted_clean_latent_indices, (0, 1, 3))
        self.assertEqual(spec.num_ref_latents, 1)
        self.assertEqual(spec.num_cond_latents, 3)

    def test_latent_bookkeeping_rejects_invalid_indices(self):
        with self.assertRaisesRegex(ValueError, "generated_latent_count"):
            build_latent_bookkeeping_spec(generated_latent_count=-1)
        with self.assertRaisesRegex(ValueError, "clean_latent_indices"):
            build_latent_bookkeeping_spec(clean_latent_indices=(-1,), generated_latent_count=1)

    def test_audio_window_payload_gates_vae_overlap_metadata(self):
        payload = build_avatar_audio_payload(full_audio_emb=FakeIndexableAudio(), num_segments=2)

        with self.assertRaisesRegex(ValueError, "prev_images and vae must be provided together"):
            build_audio_window_payload(
                payload,
                frames_processed=93,
                num_frames=93,
                overlap=13,
                prev_images=object(),
            )

        window_payload = build_audio_window_payload(
            payload,
            frames_processed=93,
            num_frames=93,
            overlap=13,
            prev_images=object(),
            vae=object(),
        )

        self.assertEqual(window_payload["overlap_source"], "vae_reencode_available")

    def test_audio_window_rejects_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "frames_processed"):
            build_audio_window_spec(frames_processed=-1)
        with self.assertRaisesRegex(ValueError, "if_not_enough_audio"):
            build_audio_window_spec(if_not_enough_audio="loop")

    def test_output_tensor_contract_accepts_image_tensor(self):
        self.assertEqual(
            validate_output_tensor_contract(FakeTensor((93, 480, 832, 3)), expected_frames=93),
            (93, 480, 832, 3),
        )

    def test_output_tensor_contract_rejects_bad_channels(self):
        with self.assertRaisesRegex(ValueError, "channel count"):
            validate_output_tensor_contract(FakeTensor((93, 480, 832, 1)), expected_frames=93)

    def test_output_tensor_contract_rejects_bad_frame_count(self):
        with self.assertRaisesRegex(ValueError, "frame count"):
            validate_output_tensor_contract(FakeTensor((92, 480, 832, 3)), expected_frames=93)


if __name__ == "__main__":
    unittest.main()
