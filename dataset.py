import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from digits_autoencoder import DigitsAutoencoder

MAX_QUBITS_AVAILABLE = 16


class Dataset:
    def __init__(self, dataset, encoder_path="digits_encoder.pt", hidden_dim=MAX_QUBITS_AVAILABLE):
        self.dataset = dataset
        self.scaler = MinMaxScaler()
        self.original_features = self.scaler.fit_transform(dataset.data)
        self.labels = dataset.target
        self.encoder_path = encoder_path
        self.encoder = None

        self.features = self._maybe_encode_features(hidden_dim)

    def _maybe_encode_features(self, hidden_dim):
        input_dim = self.original_features.shape[1]
        if input_dim <= MAX_QUBITS_AVAILABLE:
            print(f"[INFO] {input_dim} features ≤ {MAX_QUBITS_AVAILABLE}. Skipping encoding.")
            return self.original_features

        print(f"[INFO] {input_dim} features > {MAX_QUBITS_AVAILABLE}. Using encoder.")

        self.encoder = DigitsAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)

        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(
                f"[ERROR] Encoder file not found at {self.encoder_path}. "
                f"Train the encoder using train_encoder.py first."
            )

        self.encoder.load_state_dict(torch.load(self.encoder_path))
        self.encoder.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(self.original_features, dtype=torch.float32)
            compressed = self.encoder.encode(X_tensor)
            return compressed.numpy()

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def get_num_features(self):
        return self.features.shape[1]

    def split_dataset(self):
        return train_test_split(self.features, self.labels, train_size=0.8)
