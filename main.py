from sklearn.datasets import load_iris

from controller_trainer import DQNAgent
from dataset import Dataset
from environment import QuantCircuitEnv

iris_data = load_iris()
dataset = Dataset(iris_data)
env = QuantCircuitEnv(4, dataset)
agent = DQNAgent(env, env.max_length, env.action_space.n)
agent.learn()
