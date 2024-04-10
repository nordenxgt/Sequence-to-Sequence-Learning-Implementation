import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_vocab: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_vocab, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        _, (hidden, cell) = self.lstm(embedded) 
        return hidden, cell