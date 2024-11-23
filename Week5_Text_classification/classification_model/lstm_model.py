import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=512, num_layers=2, dropout=0.3, num_classes=2,bidirectional=True):
        super(LSTM, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # LSTM layer
        self.model = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout = dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, texts, attention_mask=None, token_type_ids=None):

        texts = self.embedding(texts)  # [batch_size, seq_length] -> [batch_size, seq_length, embedding_dim]
        output, (hidden_state, _) = self.model(texts)
        final_output = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)  # [batch_size, hidden_size*2]
        output = self.classifier(final_output)  # [batch_size, num_classes]

        return output