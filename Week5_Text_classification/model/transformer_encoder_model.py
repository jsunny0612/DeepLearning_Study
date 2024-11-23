import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return self.encoding[:, :x.size(1), :]

class PositionWiseFCFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, head, d_ff, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, head, dropout=dropout, batch_first=True)
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFCFeedForwardNetwork(d_model, d_ff)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, padding_mask):
        # residual connection을 위해 잠시 담아둔다.
        residual = x

        # 1. multi-head attention (self attention)
        x, attention_score = self.attention(x, x, x, key_padding_mask=padding_mask)

        # 2. add & norm
        x = self.dropout(x) + residual
        x = self.layerNorm1(x)

        residual = x

        # 3. feed-forward network
        x = self.ffn(x)

        # 4. add & norm
        x = self.dropout(x) + residual
        x = self.layerNorm2(x)

        return x, attention_score

class Encoder(nn.Module):
    def __init__(self, n_input_vocab, d_model, head, d_ff, max_len, padding_idx, dropout, n_layers, device):
        super().__init__()

        # Embedding
        self.input_emb = nn.Embedding(n_input_vocab, d_model, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=dropout)

        # n개의 encoder layer를 list에 담기
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, head=head, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, padding_mask):
        # 1. 입력에 대한 input embedding, positional encoding 생성
        input_emb = self.input_emb(x)
        pos_encoding = self.pos_encoding(x)

        # 2. add & dropout
        x = self.dropout(input_emb + pos_encoding)

        # 3. n번 EncoderLayer 반복하기
        for encoder_layer in self.encoder_layers:
            x, attention_score = encoder_layer(x, padding_mask)

        return x