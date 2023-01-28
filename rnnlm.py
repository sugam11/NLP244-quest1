import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        in_embedding_dim,
        n_hidden,
        n_layers,
        dropout=0.5,
        rnn_type="elman",  # can be elman, lstm, gru
        bi_direction=False,
    ):
        super(RNNModel, self).__init__()

        if rnn_type == "elman":
            self.rnn = nn.RNN(
                in_embedding_dim,
                n_hidden,
                n_layers,
                nonlinearity="tanh",
                dropout=dropout,
                bidirectional=bi_direction,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                in_embedding_dim,
                n_hidden,
                n_layers,
                dropout=dropout,
                bidirectional=bi_direction,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                in_embedding_dim,
                n_hidden,
                n_layers,
                dropout=dropout,
                bidirectional=bi_direction,
            )
        
        self.in_embedder = nn.Embedding(vocab_size, in_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.Linear(n_hidden * 2, vocab_size) if bi_direction else nn.Linear(n_hidden, vocab_size)
        self.init_weights()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.bi_direction = bi_direction

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.in_embedder.weight, -initrange, initrange)
        nn.init.zeros_(self.pooling.bias)
        nn.init.uniform_(self.pooling.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.dropout(self.in_embedder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout(output)
        pooled = self.pooling(output)
        pooled = pooled.view(-1, self.vocab_size)
        return F.log_softmax(pooled, dim=1), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        first_dim = self.n_layers * 2 if self.bi_direction else self.n_layers
        if self.rnn_type == "lstm":
            return (
                    weight.new_zeros(first_dim, batch_size, self.n_hidden), 
                    weight.new_zeros(first_dim, batch_size, self.n_hidden)
                    )
        return weight.new_zeros(first_dim, batch_size, self.n_hidden)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            model = torch.load(f)
            model.rnn.flatten_parameters()
            return model
