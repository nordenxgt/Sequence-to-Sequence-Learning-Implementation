import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, output_vocab: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: int):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_vocab)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return self.fc(output)