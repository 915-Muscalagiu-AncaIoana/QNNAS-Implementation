import torch.nn as nn
from abc import ABC, abstractmethod


class BaseAutoencoder(nn.Module, ABC):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = self._build_encoder(input_dim, hidden_dim)
        self.decoder = self._build_decoder(hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    @abstractmethod
    def _build_encoder(self, input_dim, hidden_dim):
        pass

    @abstractmethod
    def _build_decoder(self, hidden_dim, output_dim):
        pass
