import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
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
        self.input_parameters = ParameterVector('sigma', length=num_qubits)
        self.parameters = ParameterVector('theta', length=num_qubits * 2)
        self.qc = QuantumCircuit(num_qubits)

        for i in range(num_qubits):
            self.qc.rx(self.input_parameters[i], i)
            self.qc.ry(self.parameters[2 * i], i)
            self.qc.rz(self.parameters[2 * i + 1], i)

        self.qnn = EstimatorQNN(
            circuit=self.qc,
            input_params=list(self.input_parameters),
            weight_params=list(self.parameters),
        )
        self.torch_qnn = TorchConnector(self.qnn)

    def forward(self, x):
        return self.torch_qnn(x)


class QuantumQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=16):
        super(QuantumQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.quantum_layer = QuantumLayer(hidden_dim)
        self.fc2 = nn.Linear(1, action_dim)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.quantum_layer(x)
        return self.fc2(x)
