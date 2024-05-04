import numpy as np
import gym
from gym.vector.utils import spaces
from matplotlib import pyplot as plt
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from gym import utils
import qiskit as qk
from qiskit_machine_learning.neural_networks import EstimatorQNN

from child_net_trainer import ChildNetTrainer

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)
action_to_gate = {
    1: 'rx', 2: 'ry', 3: 'rz', 4: 'cx', 5: 'cy', 6: 'cz'
}


class QuantCircuitEnv(gym.Env):
    def __init__(self, max_length, dataset):
        self.num_qubits = dataset.get_num_features()
        self.state = []
        self.possible_rotation_gates = ['rx', 'ry', 'rx']
        self.possible_entaglement_gates = ['cx', 'cz']
        self.action_space = spaces.Discrete(5)
        self.max_length = max_length
        self.done = False
        self.dataset = dataset

    def get_observation_info(self):
        obs = self.state
        self.done = len(self.state) == self.max_length
        reward = self._compute_reward()
        info = {
            'architecture_length': len(self.state),
            'max_length': self.max_length,
            'state_chain': self.state,
        }
        return obs, reward, self.done, info

    def step(self, action):
        if self.done:
            return

        self.state.append(action)
        observation, reward, self.done, info = self.get_observation_info()

        return observation, reward, self.done, info

    def _compute_reward(self):
        self.build_final_classifier()
        gates = [action_to_gate.get(action + 1) for action in self.state]
        trainer = ChildNetTrainer(self.dataset, self.qnn, gates)
        return trainer.train_child_net()

    def reset(self):
        self.state = []
        self.done = False
        return self.state

    def render(self):
        self.build_final_classifier()
        print(self.qnn.circuit)

    def state_to_ansatz(self):
        gates = [action_to_gate.get(action + 1) for action in self.state]
        feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=1)
        ansatz = TwoLocal(self.num_qubits, self._filter_rotation_gates(gates), self._filter_entanglement_gates(gates), 'circular',
                          insert_barriers=True,
                          skip_final_rotation_layer=True)
        return feature_map, ansatz

    def build_circuit(self):
        qc = qk.QuantumCircuit(self.num_qubits)
        feature_map, ansatz = self.state_to_ansatz()
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        return qc, feature_map.parameters, ansatz.parameters

    def build_final_classifier(self):
        qc, input_param, param = self.build_circuit()
        self.qnn = EstimatorQNN(
            circuit=qc,
            input_params=input_param,
            weight_params=param
        )

    def seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        np.random.seed(seed)
        return [seed]

    @staticmethod
    def _filter_rotation_gates(gates):
        return [gate for gate in gates if gate[0] == 'r']

    @staticmethod
    def _filter_entanglement_gates(gates):
        return [gate for gate in gates if gate[0] == 'c']
