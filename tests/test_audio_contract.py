import unittest
from types import SimpleNamespace

from LongCat_Video.audio_contract import (
    AVATAR_AUDIO_LAYERS,
    AVATAR_AUDIO_STRIDE,
    AVATAR_NUM_COND_FRAMES,
    AVATAR_NUM_FRAMES,
    AVATAR_SAVE_FPS,
    AUDIO_TYPE_ADD,
    AUDIO_TYPE_PARA,
    AVATAR_AUDIO_PAYLOAD_TYPE,
    LONGCAT_AVATAR_WHISPER_ENCODER,
    MAX_ADVANCED_SPEAKER_TRACKS,
    build_avatar_audio_payload,
    build_multi_speaker_audio_payload,
    calculate_generate_duration,
    calculate_num_segments_for_audio_duration,
    calculate_num_segments_for_prepared_audio_lengths,
    calculate_num_segments_for_sample_count,
    calculate_source_sample_count_for_prepared_audio_lengths,
    calculate_target_output_frames_for_sample_count,
    ensure_mono_waveform_array,
    normalize_longcat_avatar_whisper_state_dict,
    prepared_multi_audio_length,
    validate_avatar_audio_payload_metadata,
    validate_audio_conditioning_payload,
    validate_audio_embedding,
    validate_longcat_avatar_whisper_load_result,
    validate_longcat_avatar_whisper_model_name,
    validate_matching_audio_embedding_shapes,
    validate_multi_audio_lengths,
)
from LongCat_Video.audio_crop import crop_audio_payload, parse_audio_crop_time


class FakeTensor:
    def __init__(self, shape, *, finite=True):
        self.shape = tuple(shape)
        self.finite = finite

    def isfinite(self):
        return FakeFinite(self.finite)


class FakeFinite:
    def __init__(self, finite):
        self.finite = finite

    def all(self):
        return self

    def item(self):
        return self.finite


class FakeWaveform:
    ndim = 2

    def __init__(self):
        self.mean_axis = None

    def mean(self, axis):
        self.mean_axis = axis
        return "mono"


class FakeAudioWaveform:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.last_key = None

    def __getitem__(self, key):
        self.last_key = key
        sample_slice = key[-1] if isinstance(key, tuple) else key
        start = 0 if sample_slice.start is None else sample_slice.start
        stop = self.shape[-1] if sample_slice.stop is None else sample_slice.stop
        return FakeAudioWaveform((*self.shape[:-1], max(0, stop - start)))


def _embedding(frames=100, layers=AVATAR_AUDIO_LAYERS, width=1280, *, finite=True):
    return FakeTensor((frames, layers, width), finite=finite)


class AudioContractTests(unittest.TestCase):
    def test_audio_crop_time_parser_accepts_seconds_and_clock_strings(self):
        self.assertEqual(parse_audio_crop_time("10", field_name="start_time"), 10.0)
        self.assertEqual(parse_audio_crop_time("1:02", field_name="start_time"), 62.0)
        self.assertEqual(parse_audio_crop_time("1:02:03.5", field_name="start_time"), 3723.5)

    def test_audio_crop_time_parser_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "start_time"):
            parse_audio_crop_time("-1", field_name="start_time")
        with self.assertRaisesRegex(ValueError, "end_time"):
            parse_audio_crop_time("bad", field_name="end_time")

    def test_audio_crop_payload_preserves_sample_rate_and_dimensions(self):
        audio = {"waveform": FakeAudioWaveform((1, 2, 5000)), "sample_rate": 100}

        result = crop_audio_payload(audio, start_time="0:10", end_time="0:30")

        self.assertEqual(result["sample_rate"], 100)
        self.assertEqual(result["waveform"].shape, (1, 2, 2000))
        self.assertEqual(audio["waveform"].last_key[-1], slice(1000, 3000, None))

    def test_audio_crop_payload_supports_seconds_only_input(self):
        result = crop_audio_payload(
            {"waveform": FakeAudioWaveform((1, 1, 5000)), "sample_rate": 100},
            start_time="10",
            end_time="30",
        )

        self.assertEqual(result["waveform"].shape, (1, 1, 2000))

    def test_audio_crop_payload_clamps_end_to_full_audio_length(self):
        result = crop_audio_payload(
            {"waveform": FakeAudioWaveform((1, 1, 2000)), "sample_rate": 100},
            start_time="0:05",
            end_time="0:30",
        )

        self.assertEqual(result["waveform"].shape, (1, 1, 1500))

    def test_audio_crop_payload_rejects_reversed_range(self):
        with self.assertRaisesRegex(ValueError, "start_time"):
            crop_audio_payload(
                {"waveform": FakeAudioWaveform((1, 1, 5000)), "sample_rate": 100},
                start_time="0:30",
                end_time="0:10",
            )

    def test_audio_crop_payload_allows_empty_boundary_crop(self):
        result = crop_audio_payload(
            {"waveform": FakeAudioWaveform((1, 1, 500)), "sample_rate": 100},
            start_time="0:10",
            end_time="0:30",
        )

        self.assertEqual(result["waveform"].shape, (1, 1, 0))

    def test_longcat_avatar_whisper_model_accepts_expected_checkpoint(self):
        self.assertEqual(
            validate_longcat_avatar_whisper_model_name(LONGCAT_AVATAR_WHISPER_ENCODER),
            LONGCAT_AVATAR_WHISPER_ENCODER,
        )

    def test_longcat_avatar_whisper_model_rejects_wav2vec_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "LongCat Avatar Whisper.*whisper-large-v3"):
            validate_longcat_avatar_whisper_model_name("wav2vec2-chinese_base_fp16.safetensors")

    def test_whisper_state_dict_strips_transformers_model_prefix(self):
        state_dict = {
            "model.encoder.conv1.weight": object(),
            "model.decoder.embed_tokens.weight": object(),
        }

        normalized = normalize_longcat_avatar_whisper_state_dict(state_dict)

        self.assertEqual(
            tuple(normalized.keys()),
            ("encoder.conv1.weight", "decoder.embed_tokens.weight"),
        )
        self.assertIs(normalized["encoder.conv1.weight"], state_dict["model.encoder.conv1.weight"])

    def test_whisper_state_dict_keeps_unprefixed_keys(self):
        state_dict = {"encoder.conv1.weight": object()}

        self.assertIs(normalize_longcat_avatar_whisper_state_dict(state_dict), state_dict)

    def test_whisper_state_dict_rejects_mixed_prefixes(self):
        with self.assertRaisesRegex(ValueError, "mixes prefixed and unprefixed"):
            normalize_longcat_avatar_whisper_state_dict(
                {
                    "model.encoder.conv1.weight": object(),
                    "encoder.conv2.weight": object(),
                }
            )

    def test_whisper_load_result_rejects_silent_state_dict_mismatch(self):
        result = SimpleNamespace(
            missing_keys=["encoder.conv1.weight"],
            unexpected_keys=["model.encoder.conv1.weight"],
        )

        with self.assertRaisesRegex(ValueError, "LongCat Avatar Whisper state dict mismatch"):
            validate_longcat_avatar_whisper_load_result(result)

    def test_whisper_load_result_allows_exact_match(self):
        validate_longcat_avatar_whisper_load_result(SimpleNamespace(missing_keys=[], unexpected_keys=[]))

    def test_generate_duration_matches_official_formula(self):
        duration = calculate_generate_duration(AVATAR_SAVE_FPS, 3)
        expected = AVATAR_NUM_FRAMES / AVATAR_SAVE_FPS
        expected += 2 * (AVATAR_NUM_FRAMES - AVATAR_NUM_COND_FRAMES) / AVATAR_SAVE_FPS

        self.assertAlmostEqual(duration, expected)

    def test_audio_duration_calculates_minimum_covering_segments(self):
        self.assertEqual(calculate_num_segments_for_audio_duration(18.0, AVATAR_SAVE_FPS), 6)
        self.assertLess(calculate_generate_duration(AVATAR_SAVE_FPS, 5), 18.0)
        self.assertGreaterEqual(calculate_generate_duration(AVATAR_SAVE_FPS, 6), 18.0)

    def test_sample_count_calculates_segments_after_resample(self):
        sample_count = 18 * 16000

        self.assertEqual(calculate_num_segments_for_sample_count(sample_count, 16000, AVATAR_SAVE_FPS), 6)

    def test_ten_second_audio_targets_source_duration_frames_not_segment_envelope(self):
        sample_count = 10 * 16000

        self.assertEqual(calculate_num_segments_for_sample_count(sample_count, 16000, AVATAR_SAVE_FPS), 3)
        self.assertEqual(calculate_target_output_frames_for_sample_count(sample_count, 16000, AVATAR_SAVE_FPS), 250)

    def test_audio_slightly_over_ten_seconds_trims_four_segment_envelope(self):
        sample_count = int(10.2 * 16000)

        self.assertEqual(calculate_num_segments_for_sample_count(sample_count, 16000, AVATAR_SAVE_FPS), 4)
        self.assertEqual(calculate_target_output_frames_for_sample_count(sample_count, 16000, AVATAR_SAVE_FPS), 255)

    def test_prepared_lengths_calculate_single_and_para_segments(self):
        sample_count = 18 * 16000

        self.assertEqual(
            calculate_num_segments_for_prepared_audio_lengths(
                sample_count,
                16000,
                AVATAR_SAVE_FPS,
                audio_type=AUDIO_TYPE_PARA,
            ),
            6,
        )
        self.assertEqual(
            calculate_num_segments_for_prepared_audio_lengths(
                sample_count,
                16000,
                AVATAR_SAVE_FPS,
                audio_type=AUDIO_TYPE_PARA,
                left_sample_count=sample_count,
            ),
            6,
        )

    def test_prepared_lengths_calculate_add_segments_from_combined_audio(self):
        nine_seconds = 9 * 16000

        self.assertEqual(
            calculate_num_segments_for_prepared_audio_lengths(
                nine_seconds,
                16000,
                AVATAR_SAVE_FPS,
                audio_type=AUDIO_TYPE_ADD,
                left_sample_count=nine_seconds,
            ),
            6,
        )
        self.assertEqual(
            calculate_source_sample_count_for_prepared_audio_lengths(
                nine_seconds,
                audio_type=AUDIO_TYPE_ADD,
                left_sample_count=nine_seconds,
            ),
            18 * 16000,
        )

    def test_prepared_lengths_reject_unequal_para_segments(self):
        with self.assertRaisesRegex(ValueError, "equal-length"):
            calculate_num_segments_for_prepared_audio_lengths(
                18 * 16000,
                16000,
                AVATAR_SAVE_FPS,
                audio_type=AUDIO_TYPE_PARA,
                left_sample_count=17 * 16000,
            )

    def test_segment_calculation_rejects_invalid_audio_length(self):
        with self.assertRaisesRegex(ValueError, "audio_duration"):
            calculate_num_segments_for_audio_duration(0, AVATAR_SAVE_FPS)
        with self.assertRaisesRegex(ValueError, "sample_count"):
            calculate_num_segments_for_sample_count(0, 16000, AVATAR_SAVE_FPS)

    def test_para_rejects_unequal_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal-length"):
            validate_multi_audio_lengths(100, 120, AUDIO_TYPE_PARA)

    def test_add_prepared_length_allows_unequal_inputs(self):
        length = prepared_multi_audio_length(
            100,
            40,
            generate_duration=0.5,
            sample_rate=100,
            audio_type=AUDIO_TYPE_ADD,
        )

        self.assertEqual(length, 140)

    def test_prepared_length_pads_to_target_duration(self):
        length = prepared_multi_audio_length(
            10,
            10,
            generate_duration=0.5,
            sample_rate=100,
            audio_type=AUDIO_TYPE_PARA,
        )

        self.assertEqual(length, 50)

    def test_mono_conversion_averages_channel_first_waveform(self):
        waveform = FakeWaveform()

        self.assertEqual(ensure_mono_waveform_array(waveform), "mono")
        self.assertEqual(waveform.mean_axis, 0)

    def test_audio_embedding_validates_rank_layers_and_width(self):
        self.assertEqual(validate_audio_embedding(_embedding(), "full_audio_emb"), (100, 5, 1280))

    def test_audio_embedding_rejects_wrong_layer_count(self):
        with self.assertRaisesRegex(ValueError, "layer dimension"):
            validate_audio_embedding(_embedding(layers=4), "full_audio_emb")

    def test_audio_embedding_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            validate_audio_embedding(_embedding(finite=False), "full_audio_emb")

    def test_multi_embedding_shapes_must_match(self):
        with self.assertRaisesRegex(ValueError, "shape must match"):
            validate_matching_audio_embedding_shapes(
                ("left_full_audio_emb", _embedding(frames=100)),
                ("full_audio_emb", _embedding(frames=101)),
            )

    def test_audio_conditioning_payload_validates_single(self):
        validate_audio_conditioning_payload(
            {
                "full_audio_emb": _embedding(),
                "num_segments": 1,
                "audio_stride": AVATAR_AUDIO_STRIDE,
            }
        )

    def test_audio_conditioning_payload_validates_multi_background(self):
        validate_audio_conditioning_payload(
            {
                "full_audio_emb": _embedding(),
                "left_full_audio_emb": _embedding(),
                "back_full_audio_emb": _embedding(),
                "use_background_silent_audio": True,
                "num_segments": 2,
                "audio_stride": AVATAR_AUDIO_STRIDE,
            }
        )

    def test_audio_conditioning_rejects_missing_background(self):
        with self.assertRaisesRegex(ValueError, "back_full_audio_emb"):
            validate_audio_conditioning_payload(
                {
                    "full_audio_emb": _embedding(),
                    "left_full_audio_emb": _embedding(),
                    "use_background_silent_audio": True,
                    "num_segments": 1,
                    "audio_stride": AVATAR_AUDIO_STRIDE,
                }
            )

    def test_build_avatar_audio_payload_validates_single_reusable_payload(self):
        full = _embedding(frames=120)
        payload = build_avatar_audio_payload(
            full_audio_emb=full,
            num_segments=2,
            save_fps=AVATAR_SAVE_FPS,
            audio_type=AUDIO_TYPE_PARA,
        )

        self.assertEqual(payload["payload_type"], AVATAR_AUDIO_PAYLOAD_TYPE)
        self.assertEqual(payload["audio_stride"], AVATAR_AUDIO_STRIDE)
        self.assertEqual(payload["speaker_roles"], ("primary",))
        self.assertEqual(payload["audio_features"], (full,))
        self.assertIsNone(payload["target_output_frames"])
        validate_audio_conditioning_payload(payload)
        validate_avatar_audio_payload_metadata(payload)

    def test_build_avatar_audio_payload_keeps_target_output_frame_count(self):
        payload = build_avatar_audio_payload(
            full_audio_emb=_embedding(frames=333),
            num_segments=4,
            save_fps=AVATAR_SAVE_FPS,
            audio_type=AUDIO_TYPE_PARA,
            target_output_frames=250,
        )

        self.assertEqual(payload["target_output_frames"], 250)
        validate_audio_conditioning_payload(payload)

    def test_audio_payload_rejects_target_frames_beyond_segment_coverage(self):
        with self.assertRaisesRegex(ValueError, "target_output_frames exceeds"):
            build_avatar_audio_payload(
                full_audio_emb=_embedding(frames=333),
                num_segments=4,
                target_output_frames=334,
            )

    def test_build_avatar_audio_payload_validates_dual_reusable_payload(self):
        left = _embedding()
        right = _embedding()

        payload = build_avatar_audio_payload(
            full_audio_emb=right,
            left_full_audio_emb=left,
            num_segments=1,
            save_fps=AVATAR_SAVE_FPS,
            audio_type=AUDIO_TYPE_ADD,
            left_person_bbox=[0, 0, 10, 10],
            right_person_bbox=[10, 10, 20, 20],
        )

        self.assertEqual(payload["speaker_roles"], ("left", "right"))
        self.assertEqual(payload["audio_features"], (left, right))
        self.assertEqual(payload["boxes"]["right"], [10, 10, 20, 20])
        validate_audio_conditioning_payload(payload)

    def test_audio_payload_rejects_invalid_stride(self):
        with self.assertRaisesRegex(ValueError, "audio_stride"):
            build_avatar_audio_payload(
                full_audio_emb=_embedding(),
                num_segments=1,
                audio_stride=2,
            )

    def test_audio_payload_rejects_invalid_shape(self):
        with self.assertRaisesRegex(ValueError, "layer dimension"):
            build_avatar_audio_payload(
                full_audio_emb=_embedding(layers=4),
                num_segments=1,
            )

    def test_audio_payload_rejects_non_finite_embeddings(self):
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            build_avatar_audio_payload(
                full_audio_emb=_embedding(finite=False),
                num_segments=1,
            )

    def test_legacy_audio_conditioning_payload_remains_valid_without_payload_type(self):
        validate_audio_conditioning_payload(
            {
                "full_audio_emb": _embedding(),
                "num_segments": 1,
                "audio_stride": AVATAR_AUDIO_STRIDE,
            }
        )

    def test_multi_speaker_payload_supports_bounded_advanced_tracks(self):
        speakers = {
            "speaker_1": _embedding(),
            "speaker_2": _embedding(),
            "speaker_3": _embedding(),
            "speaker_4": _embedding(),
        }

        payload = build_multi_speaker_audio_payload(
            speakers,
            num_segments=1,
            audio_type=AUDIO_TYPE_PARA,
            boxes={"speaker_1": [0, 0, 10, 10]},
            masks={"speaker_2": "mask"},
        )

        self.assertEqual(len(payload["speaker_roles"]), MAX_ADVANCED_SPEAKER_TRACKS)
        self.assertEqual(payload["speaker_mode"], "advanced_experimental")
        self.assertEqual(payload["audio_type"], AUDIO_TYPE_PARA)
        validate_audio_conditioning_payload(payload)

    def test_multi_speaker_payload_keeps_dual_tracks_official_by_default(self):
        payload = build_multi_speaker_audio_payload(
            {"left": _embedding(), "right": _embedding()},
            num_segments=1,
            audio_type=AUDIO_TYPE_ADD,
        )

        self.assertEqual(payload["speaker_roles"], ("left", "right"))
        self.assertEqual(payload["speaker_mode"], "official_dual")
        self.assertEqual(payload["left_full_audio_emb"], payload["audio_features"][0])
        self.assertEqual(payload["full_audio_emb"], payload["audio_features"][1])

    def test_multi_speaker_payload_rejects_too_many_tracks(self):
        speakers = {f"speaker_{idx}": _embedding() for idx in range(MAX_ADVANCED_SPEAKER_TRACKS + 1)}

        with self.assertRaisesRegex(ValueError, "at most"):
            build_multi_speaker_audio_payload(speakers, num_segments=1)

    def test_multi_speaker_payload_rejects_mismatched_feature_shapes(self):
        with self.assertRaisesRegex(ValueError, "shape must match"):
            build_multi_speaker_audio_payload(
                {"left": _embedding(frames=100), "right": _embedding(frames=101)},
                num_segments=1,
            )

    def test_multi_speaker_payload_rejects_invalid_masks_and_boxes(self):
        with self.assertRaisesRegex(TypeError, "masks"):
            build_multi_speaker_audio_payload(
                {"left": _embedding(), "right": _embedding()},
                num_segments=1,
                masks=["bad"],
            )
        with self.assertRaisesRegex(TypeError, "boxes"):
            build_multi_speaker_audio_payload(
                {"left": _embedding(), "right": _embedding()},
                num_segments=1,
                boxes=["bad"],
            )


if __name__ == "__main__":
    unittest.main()
