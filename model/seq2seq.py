import random

import torch
from torch import nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        assert encoder.hidden_dim == decoder.hidden_dim
        assert encoder.num_layers == decoder.num_layers
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self._init_weights()

    def _init_weights(self):
        for _, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src: torch.Tensor, target: torch.Tensor, tf_ratio: float) -> torch.Tensor:
        target_length = target.shape[1]

        outputs = torch.zeros(target.shape[0], target_length, self.decoder.output_vocab).to(self.device)
        hidden, cell = self.encoder(src)
        input = target[:, 0]

        for t in range(1, target_length):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            input = target[:, t] if random.random() < tf_ratio else output.argmax(1)

        return outputs