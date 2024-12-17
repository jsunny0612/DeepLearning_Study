''' Define the Transformer model '''

import torch
import torch.nn as nn
import numpy as np
# from Week7_Transformer_Implementation.model.Layer import EncoderLayer,DecoderLayer
from Week7_Transformer_Implementation.model.Layer import *
# from model.Layer import EncoderLayer,DecoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        pos = torch.arange(max_len, device=device).unsqueeze(1)  # [0, 1, 2, ..., max-len -1 ] == (max_len, 1)
        even_indices = torch.arange(0, d_model, 2, device=device)  # [0, 2, 4, ..., d_model-2] == 2i
        div_term = torch.pow(10000, even_indices / d_model)

        # 위치 인코딩 계산 (짝수: sin, 홀수: cos)
        pe = torch.zeros(max_len, d_model, device=device)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(pos / div_term)  # 짝수
        pe[:, 1::2] = torch.cos(pos / div_term)  # 홀수

        self.pe = pe
        self.pe.requires_grad = False

    def forward(self, x):
        seq_len = x.size(1)   # x = (batch_size,seq_len,d_model)
        return x + self.pe[:seq_len, :].unsqueeze(0).to(x.device)   # x (batch_size,seq_len,d_model) + (1,seq_len,d_model)


class Encoder(nn.Module):
    def __init__(self, n_input_vocab, d_model, head, d_ff, max_len,
                 padding_idx, dropout, n_layers, device):
        super().__init__()

        # Embedding
        self.input_emb = nn.Embedding(n_input_vocab, d_model, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=dropout)

        # n개의 encoder layer
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model= d_model, head=head, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, x, padding_mask):
        x = self.input_emb(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for encoder_layer in self.encoder_layers:
            x, attention_score = encoder_layer(x, padding_mask)

        return x


class Decoder(nn.Module):
    def __init__(self, n_output_vocab, d_model, head, d_ff, max_len, padding_idx,
                 dropout, n_layers, device):
        super().__init__()

        # output Embedding
        self.output_emb = nn.Embedding(n_output_vocab, d_model, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=dropout)

        # n 개의 decoder layer
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, head=head, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, x, memory, look_ahead_mask, padding_mask):

        x = self.output_emb(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, memory, look_ahead_mask, padding_mask)

        return x

class Transformer(nn.Module):
    def __init__(self, n_input_vocab, n_output_vocab, d_model, head, d_ff, max_len,
                 padding_idx, dropout, n_layers, device):
        super().__init__()
        self.padding_idx = padding_idx
        self.device = device

        # Encoder
        self.encoder = Encoder(n_input_vocab=n_input_vocab,
                               d_model=d_model,
                               head=head,
                               d_ff=d_ff,
                               max_len=max_len,
                               padding_idx=padding_idx,
                               dropout=dropout,
                               n_layers=n_layers,
                               device=device)

        # Decoder
        self.decoder = Decoder(n_output_vocab=n_output_vocab,
                               d_model=d_model,
                               head=head,
                               d_ff=d_ff,
                               max_len=max_len,
                               padding_idx=padding_idx,
                               dropout=dropout,
                               n_layers=n_layers,
                               device=device)

        # linear layer
        self.linear = nn.Linear(d_model, n_output_vocab)

    def forward(self, source, target):

        padding_mask = self.make_padding_mask(source, source)
        enc_dec_padding_mask = self.make_padding_mask(target, source)
        look_ahead_mask = self.make_padding_mask(target, target) * self.make_look_ahead_mask(target)

        output_encoder = self.encoder(source, padding_mask)
        output_decoder = self.decoder(target, output_encoder, look_ahead_mask, enc_dec_padding_mask)
        output = self.linear(output_decoder)

        return output

    def make_padding_mask(self, q, k):

        # q,k의 size = (batch_size, seq_len)
        _, q_seq_len = q.size()
        _, k_seq_len = k.size()

        q_mask = (q != self.padding_idx)
        q_mask = q_mask.unsqueeze(1).unsqueeze(3) # (batch_size, 1, q_seq_len, 1)
        q_mask = q_mask.repeat(1,1,1,k_seq_len)   # (batch_size, 1, q_seq_len, k_seq_len)

        k_mask = (k != self.padding_idx)
        k_mask = k_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, k_seq_len)
        k = k_mask.repeat(1,1,q_seq_len,1)   # (batch_size, 1, q_seq_len, k_seq_len)

        padding_mask = q_mask & k_mask     # (batch_size, 1, q_seq_len, k_seq_len)

        return padding_mask

    def make_look_ahead_mask(self, target):

        _, seq_len = target.size()

        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.bool().to(self.device)

        return mask





