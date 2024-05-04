import numpy as np
import torch
from qiskit_machine_learning.connectors import TorchConnector
from torch.optim import Adam


class ChildNetTrainer:
    def __init__(self, dataset, qnn, state):
        self. dataset = dataset
        self.qnn = TorchConnector(qnn)
        self.state = state
        print(f'Training Child net with arch: {self.state}')

    def train_child_net(self):
        X_train, X_test, Y_train, Y_test = self.dataset.split_dataset()
        num_qubits = self.dataset.get_num_features()
        X_train_torch = torch.tensor(X_train[:, :num_qubits], dtype=torch.float32)
        y_train_torch = torch.tensor(Y_train, dtype=torch.float32)
        optimizer = Adam(self.qnn.parameters(), lr=0.1)

        for epoch in range(2):
            self.qnn.train()
            optimizer.zero_grad()
            # Forward pass
            outputs = self.qnn(X_train_torch).reshape(-1)
            loss_train = torch.nn.functional.mse_loss(outputs, y_train_torch)
            # Backward pass and optimize
            loss_train.backward()
            optimizer.step()

            print(f'Epoch {epoch + 1}, Train Loss: {loss_train.item()}')

        # Evaluate model
        self.qnn.eval()
        X_test_torch = torch.tensor(X_test[:, :num_qubits], dtype=torch.float32)
        Y_pred = self.qnn(X_test_torch).detach().numpy().round()
        accuracy = np.mean(Y_pred.flatten() == Y_test)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        return accuracy
