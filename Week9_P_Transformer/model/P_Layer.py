''' Define the P-EncoderLayer and P-DecoderLayer '''

import torch
import torch.nn as nn
import numpy as np
from Week9_P_Transformer.model.P_SubLayers import MultiHeadAttention, FeedForwardNetwork


class EncoderLayer(nn.Module):
    def __init__(self, d_model, head, d_ff, dropout):
        super().__init__()
        self.multi_attention = MultiHeadAttention(d_model,head)
        self.layerNorm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model,d_ff)
        self.layerNorm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, padding_mask):

        residual = x

        x, attention_score = self.multi_attention(q=x, k=x, v=x, mask=padding_mask)

        x = self.dropout(x) + residual
        x = self.layerNorm1(x)

        residual = x

        x = self.ffn(x)
        x = self.dropout(x) + residual
        x = self.layerNorm2(x)

        return x, attention_score


class DecoderLayer(nn.Module):
    def __init__(self, d_model, head, d_ff, dropout):
        super().__init__()

        self.attention1 = MultiHeadAttention(d_model,head)
        self.layerNorm1 = nn.LayerNorm(d_model)

        self.attention2 = MultiHeadAttention(d_model,head)
        self.layerNorm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model,d_ff)
        self.layerNorm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, memory, look_ahead_mask, padding_mask):

        residual = x
        x, _ = self.attention1(q=x, k=x, v=x, mask=look_ahead_mask)

        x = self.dropout(x) + residual
        x = self.layerNorm1(x)

        residual = x
        x, _ = self.attention2(q=x, k=memory, v=memory, mask=padding_mask)

        x = self.dropout(x) + residual
        x = self.layerNorm2(x)

        residual = x
        x = self.ffn(x)

        x = self.dropout(x) + residual
        x = self.layerNorm3(x)

        return x

