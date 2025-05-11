import torch.nn as nn
from base_autoencoder import BaseAutoencoder

class DigitsAutoencoder(BaseAutoencoder):
    def _build_encoder(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.ReLU()
        )

    def _build_decoder(self, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )
