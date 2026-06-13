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
        global select_query_chunk_size
        from LongCat_Video.longcat_video.modules.attention_ops import (
            attention_score_buffer_bytes,
            attention_telemetry_summary,
            chunked_eager_attention,
            format_attention_telemetry,
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


if __name__ == "__main__":
    unittest.main()
