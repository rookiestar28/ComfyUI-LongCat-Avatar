import os
import tempfile
import unittest
from pathlib import Path

from LongCat_Video.video_output import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_QUALITY,
    build_video_output_plan,
    normalize_mux_audio_path,
    sanitize_video_prefix,
    save_muxed_video,
    validate_output_directory,
)


class VideoOutputTests(unittest.TestCase):
    def test_disabled_output_returns_empty_plan_without_audio(self):
        plan = build_video_output_plan(
            enabled=False,
            output_dir="/does/not/matter",
            audio_path=None,
        )

        self.assertFalse(plan.enabled)
        self.assertEqual(plan.final_path, "")

    def test_sanitizes_prefix(self):
        self.assertEqual(sanitize_video_prefix("Long Cat Avatar!"), "Long_Cat_Avatar")
        self.assertEqual(sanitize_video_prefix("..."), "longcat_avatar")

    def test_normalizes_disabled_mux_audio_sentinels(self):
        for value in (None, "", "   ", "0", " none ", "NULL"):
            with self.subTest(value=value):
                self.assertEqual(normalize_mux_audio_path(value), "")

    def test_enabled_output_requires_audio_file(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(FileNotFoundError, "mux_audio_path"):
                build_video_output_plan(
                    enabled=True,
                    output_dir=output_dir,
                    audio_path=os.path.join(output_dir, "missing.wav"),
                )

    def test_rejects_reference_output_directory(self):
        with tempfile.TemporaryDirectory() as root:
            reference_dir = Path(root) / "reference" / "out"
            reference_dir.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "reference"):
                validate_output_directory(reference_dir)

    def test_enabled_plan_stays_under_output_directory(self):
        with tempfile.TemporaryDirectory() as output_dir:
            audio_path = os.path.join(output_dir, "speech.wav")
            Path(audio_path).write_bytes(b"fake")

            plan = build_video_output_plan(
                enabled=True,
                output_dir=output_dir,
                audio_path=audio_path,
                prefix="../bad prefix",
                mode="single",
                token="fixed",
            )

            self.assertTrue(plan.enabled)
            self.assertTrue(plan.final_path.startswith(str(Path(output_dir).resolve())))
            self.assertTrue(plan.final_path.endswith(".mp4"))
            self.assertIn("bad_prefix_single_fixed", plan.final_path)

    def test_disabled_save_does_not_call_saver(self):
        calls = []

        result = save_muxed_video(
            frames=object(),
            enabled=False,
            output_dir="/does/not/matter",
            audio_path=None,
            prefix="demo",
            mode="single",
            saver=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

        self.assertEqual(result, "")
        self.assertEqual(calls, [])

    def test_enabled_save_invokes_saver_with_official_defaults(self):
        calls = []

        def fake_saver(*args, **kwargs):
            calls.append((args, kwargs))

        with tempfile.TemporaryDirectory() as output_dir:
            audio_path = os.path.join(output_dir, "speech.wav")
            Path(audio_path).write_bytes(b"fake")

            result = save_muxed_video(
                frames="frames",
                enabled=True,
                output_dir=output_dir,
                audio_path=audio_path,
                prefix="demo",
                mode="multi",
                saver=fake_saver,
            )

        self.assertTrue(result.endswith(".mp4"))
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(args[0], "frames")
        self.assertEqual(args[2], audio_path)
        self.assertEqual(kwargs["fps"], DEFAULT_VIDEO_FPS)
        self.assertEqual(kwargs["quality"], DEFAULT_VIDEO_QUALITY)


if __name__ == "__main__":
    unittest.main()
