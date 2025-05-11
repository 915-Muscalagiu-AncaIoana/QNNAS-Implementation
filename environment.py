import os
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
    def __init__(self, max_length, dataset, gates=None):
        if gates is None:
            gates = ['rx', 'ry', 'rz', 'cx', 'cy']
        self.num_qubits = dataset.get_num_features()
        self.state = [0] * max_length
        self.state_length = 0
        self.possible_rotation_gates = self._filter_rotation_gates(gates)
        self.possible_entanglement_gates = self._filter_entanglement_gates(gates)
        self.action_space = spaces.Discrete(7)
        self.max_length = max_length
        self.done = 0
        self.dataset = dataset

    def get_observation_info(self):
        obs = self.state
        if self.state_length == self.max_length:
            self.done = 1
        reward = self._compute_reward()
        info = {
            'state_length': self.state_length,
            'max_length': self.max_length,
            'state_chain': self.state,
        }
        return obs, reward, self.done, info

    def step(self, action):
        if self.done == 1:
            return

        self.state[self.state_length] = action
        self.state_length += 1
        observation, reward, self.done, info = self.get_observation_info()

        return observation, reward, self.done, info

    def _compute_reward(self):
        self.build_final_classifier()
        gates = [action_to_gate.get(action) for action in self.state if action in action_to_gate.keys()]
        trainer = ChildNetTrainer(self.dataset, self.qnn, gates)
        return trainer.train_child_net()

    def reset(self):
        self.state = [0] * self.max_length
        self.state_length = 0
        self.done = 0
        return self.state

    def render(self):
        self.build_final_classifier()
        print(self.qnn.circuit)

    def state_to_ansatz(self):
        gates = []
        for action in self.state:
            if action != 0:
                gates.append(action_to_gate.get(action))
        feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=1)
        ansatz = TwoLocal(self.num_qubits, self._filter_rotation_gates(gates), self._filter_entanglement_gates(gates),
                          'circular',
                          insert_barriers=True,
                          skip_final_rotation_layer=True)
        return feature_map, ansatz

    def build_circuit(self):
        qc = qk.QuantumCircuit(self.num_qubits)

        feature_map, ansatz = self.state_to_ansatz()
        feature_map = feature_map.decompose()
        ansatz = ansatz.decompose()

        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        # 🔍 Print and save only the ansatz
        print(ansatz.draw(output="text"))
        os.makedirs("circuits", exist_ok=True)
        filename = f"circuits/ansatz_len{self.state_length}_step{''.join(map(str, self.state[:self.state_length]))}.png"
        fig = ansatz.draw(output="mpl")
        fig.savefig(filename)
        plt.close(fig.figure)
        print(f"[INFO] Ansatz diagram saved to: {filename}")

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
