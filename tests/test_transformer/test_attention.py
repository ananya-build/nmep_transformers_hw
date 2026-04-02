from seq2seq.transformer import MultiHeadAttention

import torch

import unittest

import torch.nn.functional as F


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.embedding_dim = 32
        self.num_heads = 8
        self.qk_length = 4
        self.value_length = 4
        self.batch_size = 2
        self.seq_len = 5

        self.mha = MultiHeadAttention(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            qk_length=self.qk_length,
            value_length=self.value_length,
        )

    def test_split_heads_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.num_heads * self.qk_length)
        out = self.mha.split_heads(x, self.qk_length)
        self.assertEqual(out.shape, (self.batch_size, self.num_heads, self.seq_len, self.qk_length))

    def test_split_heads_then_combine_heads_identity(self):
        x = torch.randn(self.batch_size, self.seq_len, self.num_heads * self.qk_length)
        split = self.mha.split_heads(x, self.qk_length)
        combined = self.mha.combine_heads(split)
        self.assertTrue(torch.allclose(x, combined))

    def test_combine_heads_shape(self):
        x = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.qk_length)
        out = self.mha.combine_heads(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.num_heads * self.qk_length))

    def test_matches_pytorch_sdpa(self):
        Q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.qk_length)
        K = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.qk_length)
        V = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.value_length)

        out_custom = self.mha.scaled_dot_product_attention(Q, K, V)
        out_ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=None)

        self.assertTrue(torch.allclose(out_custom, out_ref, atol=1e-6, rtol=1e-5))

    def test_matches_pytorch_sdpa_masked(self):
        Q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.qk_length)
        K = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.qk_length)
        V = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.value_length)

        causal_mask = torch.triu(
            torch.ones(self.seq_len, self.seq_len, dtype=torch.bool),
            diagonal=1
        )

        causal_mask_pytorch = torch.tril(
            torch.ones(self.seq_len, self.seq_len, dtype=torch.bool),
            diagonal=0
        )

        out_custom = self.mha.scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        out_ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask_pytorch)

        # print(causal_mask, causal_mask_pytorch)
        # print(out_custom)
        # print(out_ref)

        self.assertTrue(torch.allclose(out_custom, out_ref, atol=1e-6, rtol=1e-5))

    def test_forward_matches_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)

        causal_mask = torch.triu(
            torch.ones(self.seq_len, self.seq_len, dtype=torch.bool),
            diagonal=1
        )

        out = self.mha(x, x, x, mask=causal_mask)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embedding_dim))

