"""SimpleRNN / LSTM / GRU classifiers.

One architecture, three recurrent cells, so the comparison isolates the cell and
nothing else: Embedding -> recurrent -> dropout -> dense -> dropout -> logits.
Padding is masked by packing the sequences, so a short complaint in a long batch
is not scored on its padding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

CELLS = {"SimpleRNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        cell: str,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        dense_dim: int = 64,
        dropout: float = 0.4,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"unknown cell {cell!r}; expected one of {list(CELLS)}")
        self.cell_name = cell
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        rnn_kwargs = dict(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        if cell == "SimpleRNN":
            rnn_kwargs["nonlinearity"] = "tanh"
        self.rnn = CELLS[cell](**rnn_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_dim, dense_dim)
        self.act = nn.ReLU()
        self.out = nn.Linear(dense_dim, num_classes)
        self.pad_idx = pad_idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        lengths = (input_ids != self.pad_idx).sum(dim=1).clamp(min=1)
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):  # LSTM returns (h_n, c_n)
            hidden = hidden[0]
        last = hidden[-1]
        x = self.act(self.dense(self.dropout(last)))
        return self.out(self.dropout(x))

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
