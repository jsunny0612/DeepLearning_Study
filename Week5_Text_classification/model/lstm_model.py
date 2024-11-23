import torch
import torch.nn as nn
import math

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_dim_h, hidden_dim_c, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_dim_h = hidden_dim_h
        self.hidden_dim_c = hidden_dim_c

        self.input_linear = nn.Linear(input_size, 4 * hidden_dim_h, bias=bias)
        self.hidden_linear = nn.Linear(hidden_dim_h, 4 * hidden_dim_h, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim_h)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, state):
        h, c = state
        gates = self.input_linear(x) + self.hidden_linear(h)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        c_next = forget_gate * c + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)

        return h_next, c_next


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim_h, hidden_dim_c, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim_h = hidden_dim_h
        self.hidden_dim_c = hidden_dim_c
        self.layer_dim = layer_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_cells = nn.ModuleList(
            [LSTMCell(embedding_dim, hidden_dim_h, hidden_dim_c, bias=bias)] +
            [LSTMCell(hidden_dim_h, hidden_dim_h, hidden_dim_c, bias=bias) for _ in range(1, layer_dim)]
        )

        self.fc = nn.Linear(hidden_dim_h, output_dim)

    def forward(self, x, state=None):
        # x의 크기: [batch_size, seq_len]
        x = self.embedding(x)  # 임베딩 후 크기: [batch_size, seq_len, embedding_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]

        seq_len, batch_size, _ = x.size()

        if state is None:
            hidden_states = [torch.zeros(batch_size, self.hidden_dim_h, device=x.device) for _ in range(self.layer_dim)]
            cell_states = [torch.zeros(batch_size, self.hidden_dim_c, device=x.device) for _ in range(self.layer_dim)]
        else:
            hidden_states, cell_states = state
            hidden_states = list(torch.unbind(hidden_states))
            cell_states = list(torch.unbind(cell_states))

        for t in range(seq_len):
            input_t = x[t]
            for layer in range(self.layer_dim):
                hidden_states[layer], cell_states[layer] = self.lstm_cells[layer](input_t, (
                hidden_states[layer], cell_states[layer]))
                input_t = hidden_states[layer]

        final_output = self.fc(input_t)

        return final_output


