import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from mx import finalize_mx_specs
from mx import mx_mapping


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        N = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]

        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        Q = Q.view(N, seq_len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(N, seq_len_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(N, seq_len_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)

        out = torch.matmul(attention, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(N, seq_len_q, self.embed_dim)

        out = self.fc_out(out)

        return out


if __name__ == '__main__':
    # Add config arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", default=512)
    parser.add_argument("--device", default='cuda')
    args = parser.parse_args()

    num_heads = 8
    batch_size = 2
    seq_len = 10

    query = torch.randn(batch_size, seq_len, args.hidden_size, device=args.device)
    key = torch.randn(batch_size, seq_len, args.hidden_size, device=args.device)
    value = torch.randn(batch_size, seq_len, args.hidden_size, device=args.device)

    mha = MultiHeadAttention(args.hidden_size, num_heads).to(args.device)
    output = mha(query, key, value)

    # Simple MX spec for MXFP6 weights+activations
    mx_specs = {
        'w_elem_format': 'fp6_e3m2',
        'a_elem_format': 'fp6_e3m2',
        'block_size': 32,
        'bfloat': 16,
        'custom_cuda': False,
        # For quantization-aware finetuning, do backward pass in FP32
        'quantize_backprop': False,
    }
    mx_specs = finalize_mx_specs(mx_specs)

    # Auto-inject MX modules and functions
    # This will replace certain torch.nn.* and torch.nn.functional.*
    # modules/functions in the global namespace!
    mx_mapping.inject_pyt_ops(mx_specs)

    mha_mx = MultiHeadAttention(args.hidden_size, num_heads).to(args.device)
    mha_mx.load_state_dict(mha.state_dict())
    output_mx = mha_mx(query, key, value)

    # for (name1, param1), (name2, param2) in zip(mha.named_parameters(), mha_mx.named_parameters()):
    #     assert name1 == name2, "Parameter names do not match"
    #     assert torch.allclose(param1, param2), f"Parameters do not match for {name1}"

    similarity = F.cosine_similarity(output.reshape(batch_size,-1), output_mx.reshape(batch_size,-1), dim=1, eps=1e-8)
    print ("mx and fp32 sim: ", similarity)
    print("DONE!")