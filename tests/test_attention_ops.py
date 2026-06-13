import unittest

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight repo-local test envs.
    torch = None
    F = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this test environment.")
class AttentionOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global attention_score_buffer_bytes
        global attention_telemetry_summary
        global chunked_eager_attention
        global format_attention_telemetry
        global mps_attention_chunk_size
        global mps_attention_debug_enabled
        global mps_attention_max_score_bytes
        global mps_attention_strategy
        global mps_memory_safe_attention
        global select_query_chunk_size
        from LongCat_Video.longcat_video.modules.attention_ops import (
            attention_score_buffer_bytes,
            attention_telemetry_summary,
            chunked_eager_attention,
            format_attention_telemetry,
            mps_attention_chunk_size,
            mps_attention_debug_enabled,
            mps_attention_max_score_bytes,
            mps_attention_strategy,
            mps_memory_safe_attention,
            select_query_chunk_size,
        )

    def test_score_buffer_bytes_uses_full_or_chunked_query_tokens(self):
        q = torch.zeros((2, 3, 5, 4), dtype=torch.float32)
        k = torch.zeros((2, 3, 7, 4), dtype=torch.float32)

        self.assertEqual(attention_score_buffer_bytes(q, k), 2 * 3 * 5 * 7 * 4)
        self.assertEqual(attention_score_buffer_bytes(q, k, chunk_size=2), 2 * 3 * 2 * 7 * 4)

    def test_select_query_chunk_size_respects_budget_bounds(self):
        q = torch.zeros((1, 4, 9, 8), dtype=torch.float32)
        k = torch.zeros((1, 4, 11, 8), dtype=torch.float32)
        bytes_per_query = 1 * 4 * 11 * 4

        self.assertEqual(select_query_chunk_size(q, k, bytes_per_query * 3), 3)
        self.assertEqual(select_query_chunk_size(q, k, bytes_per_query * 100), 9)
        self.assertEqual(select_query_chunk_size(q, k, 1), 1)

    def test_chunked_eager_attention_matches_sdpa(self):
        generator = torch.Generator(device="cpu").manual_seed(7)
        q = torch.randn((2, 3, 5, 4), generator=generator, dtype=torch.float32)
        k = torch.randn((2, 3, 7, 4), generator=generator, dtype=torch.float32)
        v = torch.randn((2, 3, 7, 6), generator=generator, dtype=torch.float32)

        expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        actual = chunked_eager_attention(q, k, v, chunk_size=2)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_chunked_eager_attention_can_select_chunk_from_budget(self):
        generator = torch.Generator(device="cpu").manual_seed(11)
        q = torch.randn((1, 2, 6, 3), generator=generator, dtype=torch.float32)
        k = torch.randn((1, 2, 8, 3), generator=generator, dtype=torch.float32)
        v = torch.randn((1, 2, 8, 3), generator=generator, dtype=torch.float32)
        budget = 1 * 2 * 2 * 8 * 4

        expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        actual = chunked_eager_attention(q, k, v, max_score_bytes=budget)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_chunked_eager_attention_rejects_masks_until_supported(self):
        q = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        k = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        v = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

        with self.assertRaisesRegex(NotImplementedError, "unmasked attention"):
            chunked_eager_attention(q, k, v, attn_mask=torch.zeros((1, 1, 2, 2)))

    def test_chunked_eager_attention_validates_qkv_shapes(self):
        q = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        k = torch.zeros((1, 1, 3, 4), dtype=torch.float32)
        v = torch.zeros((1, 1, 3, 2), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "q/k head dimensions must match"):
            chunked_eager_attention(q, k, v, chunk_size=1)

    def test_attention_telemetry_is_content_free_metadata(self):
        q = torch.tensor([[[[3.14159, 2.71828], [1.41421, 0.57721]]]], dtype=torch.float32)
        k = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        v = torch.ones((1, 1, 2, 2), dtype=torch.float32)

        summary = attention_telemetry_summary("avatar:self", q, k, v, chunk_size=1)
        line = format_attention_telemetry(summary)

        self.assertEqual(summary["q_shape"], (1, 1, 2, 2))
        self.assertEqual(summary["score_buffer_bytes"], 1 * 1 * 2 * 2 * 4)
        self.assertEqual(summary["chunk_score_buffer_bytes"], 1 * 1 * 1 * 2 * 4)
        self.assertIn("label=avatar:self", line)
        self.assertIn("q_shape=(1, 1, 2, 2)", line)
        self.assertNotIn("3.14159", line)
        self.assertNotIn("2.71828", line)

    def test_mps_attention_env_helpers_parse_budget_chunk_and_debug(self):
        environ = {
            "LONGCAT_MPS_ATTENTION_MAX_SCORE_BYTES": "1024",
            "LONGCAT_MPS_ATTENTION_CHUNK_SIZE": "3",
            "LONGCAT_MPS_ATTENTION_DEBUG": "yes",
        }

        self.assertEqual(mps_attention_max_score_bytes(environ), 1024)
        self.assertEqual(mps_attention_chunk_size(environ), 3)
        self.assertTrue(mps_attention_debug_enabled(environ))
        self.assertFalse(mps_attention_debug_enabled({"LONGCAT_MPS_ATTENTION_DEBUG": "0"}))

    def test_mps_attention_strategy_allows_only_exact_query_chunking(self):
        self.assertEqual(mps_attention_strategy({}), "query_chunk")
        self.assertEqual(mps_attention_strategy({"LONGCAT_MPS_ATTENTION_STRATEGY": "auto"}), "query_chunk")
        self.assertEqual(mps_attention_strategy({"LONGCAT_MPS_ATTENTION_STRATEGY": "query-chunk"}), "query_chunk")

        for strategy in ("temporal_window", "sliding-window", "key_chunk", "kv_chunk"):
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(NotImplementedError, "Only exact query_chunk"):
                    mps_attention_strategy({"LONGCAT_MPS_ATTENTION_STRATEGY": strategy})

        with self.assertRaisesRegex(ValueError, "Unsupported LONGCAT_MPS_ATTENTION_STRATEGY"):
            mps_attention_strategy({"LONGCAT_MPS_ATTENTION_STRATEGY": "surprise"})

    def test_mps_memory_safe_attention_preserves_cpu_sdpa_behavior(self):
        generator = torch.Generator(device="cpu").manual_seed(17)
        q = torch.randn((1, 2, 4, 3), generator=generator, dtype=torch.float32)
        k = torch.randn((1, 2, 5, 3), generator=generator, dtype=torch.float32)
        v = torch.randn((1, 2, 5, 3), generator=generator, dtype=torch.float32)

        expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        actual = mps_memory_safe_attention(q, k, v, max_score_bytes=1, debug=False)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_mps_memory_safe_attention_chunks_native_mps_when_budget_is_small(self):
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        is_available = getattr(mps_backend, "is_available", None)
        if not callable(is_available) or not is_available():
            self.skipTest("MPS backend is not available.")

        generator = torch.Generator(device="cpu").manual_seed(23)
        q_cpu = torch.randn((1, 2, 4, 3), generator=generator, dtype=torch.float32)
        k_cpu = torch.randn((1, 2, 5, 3), generator=generator, dtype=torch.float32)
        v_cpu = torch.randn((1, 2, 5, 3), generator=generator, dtype=torch.float32)
        expected = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, dropout_p=0.0)

        actual = mps_memory_safe_attention(
            q_cpu.to("mps"),
            k_cpu.to("mps"),
            v_cpu.to("mps"),
            max_score_bytes=1,
            debug=False,
        ).to("cpu")

        self.assertTrue(torch.allclose(actual, expected, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
