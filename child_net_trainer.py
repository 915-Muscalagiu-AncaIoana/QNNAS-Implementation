import numpy as np
import torch
from qiskit_machine_learning.connectors import TorchConnector
from torch.optim import Adam


class ChildNetTrainer:
    def __init__(self, dataset, qnn, state):
        self.dataset = dataset
        self.qnn = TorchConnector(qnn)
        self.state = state
        print(f"[INFO] Initialized ChildNetTrainer")
        print(f"[INFO] Architecture state: {self.state}")

    def train_child_net(self):
        print("[INFO] Splitting dataset...")
        X_train, X_test, Y_train, Y_test = self.dataset.split_dataset()

        num_qubits = self.dataset.get_num_features()
        print(f"[INFO] Number of qubits/features used: {num_qubits}")
        print(f"[INFO] Shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")

        # Convert to torch tensors
        X_train_torch = torch.tensor(X_train[:, :num_qubits], dtype=torch.float32)
        y_train_torch = torch.tensor(Y_train, dtype=torch.float32)

        print("[INFO] Converted dataset to torch tensors")
        print(f"[INFO] X_train_torch shape: {X_train_torch.shape}, y_train_torch shape: {y_train_torch.shape}")

        optimizer = Adam(self.qnn.parameters(), lr=0.1)
        print("[INFO] Initialized Adam optimizer with lr=0.1")

        for epoch in range(2):
            print(f"[INFO] Starting epoch {epoch + 1}")

            self.qnn.train()
            print("[DEBUG] Set QNN to training mode")

            optimizer.zero_grad()
            print("[DEBUG] Optimizer gradients zeroed")

            outputs = self.qnn(X_train_torch).reshape(-1)
            print(f"[DEBUG] Forward pass complete - outputs shape: {outputs.shape}")

            loss_train = torch.nn.functional.mse_loss(outputs, y_train_torch)
            print(f"[DEBUG] Computed MSE loss: {loss_train.item():.6f}")

            loss_train.backward()
            print("[DEBUG] Backward pass complete - gradients computed")

            optimizer.step()
            print("[DEBUG] Optimizer step complete - weights updated")

            print(f"[INFO] Epoch {epoch + 1} complete - Train Loss: {loss_train.item():.6f}")

        print(f"[INFO] Epoch {epoch + 1}, Train Loss: {loss_train.item():.6f}")

        print("[INFO] Training complete. Evaluating model...")

        self.qnn.eval()
        X_test_torch = torch.tensor(X_test[:, :num_qubits], dtype=torch.float32)
        Y_pred = self.qnn(X_test_torch).detach().numpy().round()
        accuracy = np.mean(Y_pred.flatten() == Y_test)

        print(f"[INFO] Predictions: {Y_pred.flatten()}")
        print(f"[INFO] True labels: {Y_test}")
        print(f"[RESULT] Accuracy: {accuracy * 100:.2f}%")

        return accuracy
