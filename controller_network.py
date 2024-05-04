import torch.nn as nn
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        return self.fc2(x)

class QuantumLayer(nn.Module):
    def __init__(self, num_qubits):
        super(QuantumLayer, self).__init__()
        self.num_qubits = num_qubits
        self.parameters = ParameterVector('theta', length=num_qubits * 3)
        self.qc = QuantumCircuit(num_qubits)

        for i in range(num_qubits):
            self.qc.rx(self.parameters[3 * i], i)
            self.qc.ry(self.parameters[3 * i + 1], i)
            self.qc.rz(self.parameters[3 * i + 2], i)

        self.qnn = EstimatorQNN(
            circuit=self.qc,
            input_params=[],
            weight_params=list(self.parameters),
        )
        self.torch_qnn = TorchConnector(self.qnn)

    def forward(self, x):
        # Since EstimatorQNN does not need input parameters, we pass only the weights
        return self.torch_qnn(x)


class QuantumQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_qubits=4):
        super(QuantumQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.quantum_layer = QuantumLayer(num_qubits)
        self.fc2 = nn.Linear(1, action_dim)  # Assuming EstimatorQNN returns a scalar

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.quantum_layer(x)
        return self.fc2(x)