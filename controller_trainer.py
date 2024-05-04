import numpy as np
import torch
from torch import optim, Tensor
from torch.optim import Adam
import torch.nn.functional as F
from controller_network import QNetwork
from replay_memory import ReplayMemory


class DQNAgent:
    def __init__(self, env, state_dim, action_dim, hidden_dim=64, learning_rate=1e-4, batch_size=64,
                 discount_rate=0.99):
        self.model = QNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_memory = ReplayMemory(10000)
        self.batch_size = batch_size
        self.env = env
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.optimizer = Adam(self.model.parameters(), lr=1e-2)

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            print(f'Epsilon: {epsilon}: Exploration')
            return np.random.randint(self.action_dim)
        else:
            print(f'Epsilon: {epsilon}: Exploitation')
            with torch.no_grad():
                Q_values = self.model(Tensor(state)).numpy()
            return np.argmax(Q_values[0])

    def play_one_step(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = env.step(action)
        self.replay_memory.memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def sequential_training_step(self):
        experiences = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences
        current_Q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_Q_values = self.model(next_states).max(1)[0]
            expected_Q_values = rewards + (1 - dones) * self.discount_rate * next_Q_values

        loss = F.mse_loss(current_Q_values, expected_Q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self):
        rewards = []
        best_score = 0
        for episode in range(100):
            epsilon = max(1 - episode / 5, 0.01)
            obs = self.env.reset()
            episode_reward = 0
            for step in range(5):
                obs, reward, done, info = self.play_one_step(self.env, obs, epsilon)
                episode_reward += reward
                if done:
                    break
            rewards.append(episode_reward)

            if episode_reward >= best_score:
                torch.save(self.model.state_dict(), './new_model_best_weights.pth')
                best_score = episode_reward

            print("\rEpisode: {}, Reward : {}, eps: {:.3f}".format(episode, episode_reward, epsilon), end="")
            if episode > 2:
                self.sequential_training_step()
        print(rewards)
        print(best_score)
