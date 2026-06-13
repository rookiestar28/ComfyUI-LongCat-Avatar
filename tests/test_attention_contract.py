import unittest
from pathlib import Path
from types import SimpleNamespace

import LongCat_Video.attention_contract as attention_contract
from LongCat_Video.attention_contract import (
    ATTENTION_MODE_AUTO,
    ATTENTION_MODE_FLASH_ATTN_2,
    ATTENTION_MODE_FLASH_ATTN_3,
    ATTENTION_MODE_SAGEATTN,
    ATTENTION_MODE_SAGEATTN_3,
    ATTENTION_MODE_SDPA,
    ATTENTION_MODE_XFORMERS,
    ATTENTION_MODES,
    MPS_SAFE_ATTENTION_MODES,
    MPS_VISIBLE_ATTENTION_MODES,
    attention_config_fallback_warnings,
    attention_diagnostic_lines,
    apply_attention_mode_to_config,
    attention_mode_config_overrides,
    inspect_attention_backend,
    normalize_attention_mode,
    validate_attention_mode_availability,
    validate_attention_mode_for_device,
)
from LongCat_Video.mps_attention_probe import (
    build_representative_attention_probe_spec,
    run_mps_attention_equivalence_probe,
)

ROOT = Path(__file__).resolve().parents[1]


class AttentionContractTests(unittest.TestCase):
    def test_attention_modes_include_optional_accelerators(self):
        self.assertEqual(
            ATTENTION_MODES,
            (
                "auto",
                "sdpa",
                "flash_attn_2",
                "flash_attn_3",
                "xformers",
                "sageattn",
                "sageattn_3",
            ),
        )

    def test_auto_preserves_existing_config(self):
        config = {
            "enable_flashattn2": True,
            "enable_xformers": False,
            "custom": "value",
        }

        self.assertEqual(apply_attention_mode_to_config(config, ATTENTION_MODE_AUTO), config)
        self.assertEqual(attention_mode_config_overrides(ATTENTION_MODE_AUTO), {})

    def test_explicit_modes_rewrite_attention_flags(self):
        cases = {
            ATTENTION_MODE_SDPA: {},
            ATTENTION_MODE_FLASH_ATTN_2: {"enable_flashattn2": True},
            ATTENTION_MODE_FLASH_ATTN_3: {"enable_flashattn3": True},
            ATTENTION_MODE_XFORMERS: {"enable_xformers": True},
            ATTENTION_MODE_SAGEATTN: {"enable_sageattn": True},
            ATTENTION_MODE_SAGEATTN_3: {"enable_sageattn3": True},
        }

        for mode, expected_enabled in cases.items():
            with self.subTest(mode=mode):
                config = apply_attention_mode_to_config(
                    {
                        "enable_flashattn2": True,
                        "enable_flashattn3": True,
                        "enable_xformers": True,
                        "enable_sageattn": True,
                        "enable_sageattn3": True,
                    },
                    mode,
                )
                expected = {
                    "enable_flashattn2": False,
                    "enable_flashattn3": False,
                    "enable_xformers": False,
                    "enable_sageattn": False,
                    "enable_sageattn3": False,
                }
                expected.update(expected_enabled)
                self.assertEqual({key: config[key] for key in expected}, expected)

    def test_normalize_rejects_unknown_mode(self):
        self.assertEqual(normalize_attention_mode(None), ATTENTION_MODE_AUTO)
        self.assertEqual(normalize_attention_mode(""), ATTENTION_MODE_AUTO)
        with self.assertRaisesRegex(ValueError, "Unsupported attention_mode"):
            normalize_attention_mode("radial_sage_attention")

    def test_explicit_missing_backend_raises_before_runtime_fallback(self):
        original_import_module = attention_contract.importlib.import_module

        def missing_import(module_name):
            raise ModuleNotFoundError(module_name)

        attention_contract.importlib.import_module = missing_import
        try:
            with self.assertRaisesRegex(RuntimeError, "attention_mode 'sageattn'.*requires"):
                validate_attention_mode_availability(ATTENTION_MODE_SAGEATTN)
            with self.assertRaisesRegex(RuntimeError, "attention_mode 'flash_attn_2'.*requires"):
                validate_attention_mode_availability(ATTENTION_MODE_FLASH_ATTN_2)
            self.assertTrue(validate_attention_mode_availability(ATTENTION_MODE_SDPA).available)
            self.assertTrue(validate_attention_mode_availability(ATTENTION_MODE_AUTO).available)
        finally:
            attention_contract.importlib.import_module = original_import_module

    def test_sageattn3_reports_incomplete_longcat_support(self):
        original_import_module = attention_contract.importlib.import_module

        def fake_import(module_name):
            if module_name == "sageattn3":
                return SimpleNamespace(sageattn3_blackwell=lambda *args, **kwargs: None)
            raise ModuleNotFoundError(module_name)

        attention_contract.importlib.import_module = fake_import
        try:
            status = inspect_attention_backend(ATTENTION_MODE_SAGEATTN_3)
            self.assertFalse(status.available)
            self.assertIn("not fully wired", status.reason)
            with self.assertRaisesRegex(RuntimeError, "SageAttention3 is not fully wired"):
                validate_attention_mode_availability(ATTENTION_MODE_SAGEATTN_3)
        finally:
            attention_contract.importlib.import_module = original_import_module

    def test_auto_mode_reports_config_fallback_warnings(self):
        original_import_module = attention_contract.importlib.import_module

        def missing_import(module_name):
            raise ModuleNotFoundError(module_name)

        attention_contract.importlib.import_module = missing_import
        try:
            warnings = attention_config_fallback_warnings(
                ATTENTION_MODE_AUTO,
                {"enable_flashattn2": True, "enable_sageattn": True},
            )
        finally:
            attention_contract.importlib.import_module = original_import_module

        self.assertEqual(len(warnings), 2)
        self.assertIn("flash_attn_2", warnings[0])
        self.assertIn("sageattn", warnings[1])
        self.assertIn("fall back to SDPA", warnings[0])

    def test_attention_diagnostics_report_requested_mode_flags_and_backend(self):
        original_import_module = attention_contract.importlib.import_module

        def fake_import(module_name):
            if module_name == "sageattention":
                return SimpleNamespace(sageattn=lambda *args, **kwargs: None)
            raise ModuleNotFoundError(module_name)

        attention_contract.importlib.import_module = fake_import
        try:
            lines = attention_diagnostic_lines(ATTENTION_MODE_SAGEATTN, {"enable_sageattn": True})
        finally:
            attention_contract.importlib.import_module = original_import_module

        joined = "\n".join(lines)
        self.assertIn("attention_mode requested: sageattn", joined)
        self.assertIn("enable_sageattn=True", joined)
        self.assertIn("available via sageattention.sageattn", joined)

    def test_mps_allows_only_explicit_sdpa_attention(self):
        self.assertEqual(MPS_SAFE_ATTENTION_MODES, (ATTENTION_MODE_SDPA,))
        self.assertEqual(MPS_VISIBLE_ATTENTION_MODES, (ATTENTION_MODE_SDPA,))
        self.assertTrue(validate_attention_mode_for_device(ATTENTION_MODE_SDPA, "mps").available)

        for mode in (
            ATTENTION_MODE_AUTO,
            ATTENTION_MODE_FLASH_ATTN_2,
            ATTENTION_MODE_FLASH_ATTN_3,
            ATTENTION_MODE_XFORMERS,
            ATTENTION_MODE_SAGEATTN,
            ATTENTION_MODE_SAGEATTN_3,
        ):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, "limited to explicit 'sdpa'"):
                    validate_attention_mode_for_device(mode, "mps")

    def test_cuda_attention_validation_preserves_existing_sdpa_behavior(self):
        self.assertTrue(validate_attention_mode_for_device(ATTENTION_MODE_SDPA, "cuda:0").available)

    def test_attention_diagnostics_include_device_dtype_and_fallback_status(self):
        lines = attention_diagnostic_lines(ATTENTION_MODE_SDPA, {}, device="mps", dtype="float16")
        joined = "\n".join(lines)

        self.assertIn("attention runtime: device=mps", joined)
        self.assertIn("dtype=float16", joined)
        self.assertIn("mps_cpu_fallback_enabled=", joined)
        self.assertIn("attention backend 'sdpa': available", joined)

    def test_mps_attention_probe_spec_and_skip_states_are_explicit(self):
        spec = build_representative_attention_probe_spec()

        self.assertEqual(spec.device, "mps")
        self.assertEqual(spec.attention_backend, "sdpa")
        self.assertEqual(spec.query_tokens, 256)
        self.assertEqual(spec.key_tokens, 256)

        fallback_result = run_mps_attention_equivalence_probe(
            spec,
            torch_module=SimpleNamespace(),
            environ={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
        )
        self.assertEqual(fallback_result.status, "skipped")
        self.assertTrue(fallback_result.cpu_fallback_enabled)
        self.assertFalse(fallback_result.native_mps)

        unavailable_result = run_mps_attention_equivalence_probe(
            spec,
            torch_module=SimpleNamespace(
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
            ),
            environ={},
        )
        self.assertEqual(unavailable_result.status, "skipped")
        self.assertIn("MPS backend is not available", unavailable_result.reason)

    def test_avatar_attention_sdpa_paths_route_through_mps_memory_safe_helper(self):
        source = (ROOT / "LongCat_Video" / "longcat_video" / "modules" / "avatar" / "attention.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("mps_memory_safe_attention as _mps_memory_safe_attention", source)
        self.assertNotIn("_sdpa_attention", source)
        self.assertIn('_mps_memory_safe_attention(q, k, v, label="avatar:self")', source)
        self.assertIn('_mps_memory_safe_attention(q, encoder_k, encoder_v, label="avatar:cross")', source)

    def test_attention_ops_expose_mps_chunk_budget_controls(self):
        source = (ROOT / "LongCat_Video" / "longcat_video" / "modules" / "attention_ops.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("LONGCAT_MPS_ATTENTION_MAX_SCORE_BYTES", source)
        self.assertIn("LONGCAT_MPS_ATTENTION_CHUNK_SIZE", source)
        self.assertIn("LONGCAT_MPS_ATTENTION_DEBUG", source)
        self.assertIn("LONGCAT_MPS_ATTENTION_STRATEGY", source)
        self.assertIn("def mps_attention_strategy", source)
        self.assertIn("def mps_memory_safe_attention", source)


if __name__ == "__main__":
    unittest.main()
