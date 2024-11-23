import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_classes=2, dropout_rate=0.3, max_len=250, num_layers=2, padding_idx=0):
        super(TransformerEncoder, self).__init__()

        # Embedding Layer
        self.input_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout=dropout_rate, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully Connected Layer for Classification
        self.fc = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 1. Embedding and Positional Encoding
        input_emb = self.input_emb(input_ids)
        pos_encoding = self.pos_encoding(input_emb)

        # 2. Apply Dropout
        x = self.dropout(input_emb + pos_encoding)

        # 3. Transformer Encoder Layers
        x = self.transformer_encoder(x)

        # 4. Pooling and Classification
        x = x.mean(dim=1)  # Mean pooling across the sequence dimension
        return self.fc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) to match batch and sequence dimensions
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Match the sequence length and embedding dimension
        return self.dropout(x)