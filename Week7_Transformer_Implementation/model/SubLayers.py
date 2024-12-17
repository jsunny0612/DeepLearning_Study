import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):

        # attn_scores = q * k^T
        attn_scores = torch.matmul(q, k.transpose(-1,-2))

        # Scaling : (q * k^T) / sqrt(d_k)
        d_k = q.size(-1)
        attn_scores = attn_scores / math.sqrt(d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax : softmax(q * k^T) / sqrt(d_k))
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Output : softmax((q * k^T) / sqrt(d_k)) * v
        output = torch.matmul(attn_scores, v)

        return output, attn_scores


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,head):
        super().__init__()

        self.d_model = d_model
        self.head = head   # head 개수
        self.d_head = d_model // head   # d_k = d_model / h)

        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        self.attention = ScaleDotProductAttention()


    def forward(self, q, k, v, mask=None):

        batch_size, seq_len, _ = q.size()  # (batch_size, seq_len, d_model)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # (batch_size, seq_lne, head, d_head) -> (batch_size, head, seq_len, d_head) -> 각 head에 대한 독립적인 연산을 위해
        q = q.view(batch_size, -1, self.head, self.d_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.head, self.d_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.head, self.d_head).transpose(1, 2)

        # Scaled Dot-Product Attention
        attention_output, attention_score = self.attention(q, k, v, mask)

        # concat
        attention_output = attention_output.transpose(1, 2)  # (batch_size, seq_len, head, d_head)
        concat_output = torch.concat(
            [attention_output[:, :, i, :] for i in range(self.head)], dim=-1
        )  # (batch_size, seq_len, d_model)

        # Multi-head Output
        output = self.w_o(concat_output)

        return output, attention_score


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))
