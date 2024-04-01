import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_vocab: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: int):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input))
        _, (hidden, cell) = self.lstm(embedded) 
        return hidden, cell