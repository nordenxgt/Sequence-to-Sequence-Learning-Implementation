import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, output_vocab: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: int):
        super().__init__()
        self.output_vocab = output_vocab
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_vocab, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_vocab)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell)) 
        output = self.fc(output.squeeze(1))
        return output, hidden, cell