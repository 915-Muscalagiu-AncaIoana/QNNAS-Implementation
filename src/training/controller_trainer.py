import shutil
from pathlib import Path

import numpy as np
import torch
from torch import optim, Tensor
from torch.optim import Adam
import torch.nn.functional as F
from controller_network import QuantumQNetwork
from replay_memory import ReplayMemory
from utils import plot_rewards
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "new_model_best_weights.pth")


class DQNAgent:
    def __init__(self, env, state_dim, action_dim, hidden_dim=16, learning_rate=1e-4, batch_size=1,
                 discount_rate=0.99, session_id=None):
        self.model = QuantumQNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_memory = ReplayMemory(10000)
        self.batch_size = batch_size
        self.env = env
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.optimizer = Adam(self.model.parameters(), lr=1e-2)
        self.session_id = session_id

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            print(f'Epsilon: {epsilon}: Exploration')
            return np.random.randint(self.action_dim)
        else:
            print(f'Epsilon: {epsilon}: Exploitation')
            with torch.no_grad():
                Q_values = self.model(Tensor(state)).numpy()
            return np.argmax(Q_values)

    def play_one_step(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = env.step(action)
        self.replay_memory.memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def sequential_training_step(self):
        experiences = self.replay_memory.sample(min(self.batch_size, len(self.replay_memory.memory)))
        states, actions, rewards, next_states, dones = zip(*experiences)

        current_Q_values = self.model(Tensor(states)).gather(min(self.batch_size, len(self.replay_memory.memory)),
                                                             torch.tensor(actions, dtype=torch.int64).unsqueeze(
                                                                 -1)).squeeze(-1)
        with torch.no_grad():
            next_Q_values = self.model(Tensor(next_states)).max(1)[0]
            expected_Q_values = torch.tensor(rewards, dtype=torch.float32) + (
                        1 - torch.tensor(dones, dtype=torch.float32)) * self.discount_rate * torch.tensor(next_Q_values,
                                                                                                          dtype=torch.float32)

        loss = F.mse_loss(current_Q_values, expected_Q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self):
        rewards = []
        best_score = float('-inf')
        for episode in range(10):
            epsilon = max(1 - episode / 10, 0.01)
            obs = self.env.reset(episode)
            max_step_reward = float('-inf')
            best_state = None
            best_accuracy = 0

            print(f"\n[EPISODE {episode}] Starting with epsilon={epsilon:.3f}")

            for step in range(5):
                obs, reward, done, info = self.play_one_step(self.env, obs, epsilon)
                penalty = 0.01 * (step + 1)
                adjusted_reward = reward - penalty

                print(f"  Step {step}: reward={reward:.4f}, penalty={penalty:.4f}, adjusted={adjusted_reward:.4f}")

                if adjusted_reward > max_step_reward:
                    max_step_reward = adjusted_reward
                    best_state = self.env.state[:self.env.state_length]
                    best_accuracy = reward
                    print(f"New best step reward: {max_step_reward:.4f}")
                    print(f"Best state: {best_state}")
                    self.env.render()

                if done:
                    print("Environment returned done=True, ending episode early.")
                    break

            rewards.append(max_step_reward)

            if best_state is not None:
                filename = f"ansatz_ep{episode}_len{len(best_state)}.png"
                source = Path.cwd() / "circuits" / str(self.env.run_id) / filename
                dest_dir = Path.cwd() / "best_architectures" / str(self.env.run_id)
                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_circuit = dest_dir / f"circuit_epoch{episode}.png"
                dest_metrics = dest_dir / f"metrics_epoch{episode}.txt"
                dest_loss_plot = dest_dir / f"loss_epoch{episode}.png"

                print(f"Moving best diagram for episode {episode}: {source} → {dest_circuit}")
                if source.exists():
                    shutil.copy2(source, dest_circuit)
                    print(f"Saved per-episode best diagram to: {dest_circuit}")
                else:
                    print(f"File not found: {source}")

                # Copy loss plot if it exists
                source_loss_plot = Path.cwd() / "losses" / str(self.env.run_id) / f"loss_epoch{episode}.png"
                if source_loss_plot.exists():
                    shutil.copy2(source_loss_plot, dest_loss_plot)
                    print(f"Loss plot copied to: {dest_loss_plot}")
                else:
                    print(f"Loss plot not found: {source_loss_plot}")

                # Save metrics file
                with open(dest_metrics, "w") as f:
                    f.write(f"{best_accuracy:.4f}\n")
                print(f"Metrics written to: {dest_metrics}")

            # Save best model overall
            if max_step_reward >= best_score:
                print(f"New overall best score! ({max_step_reward:.4f} >= {best_score:.4f})")
                best_score = max_step_reward

                best_model_path = Path.cwd() / f"models/best_model_session_{self.session_id}.pt"
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Best model saved to: {best_model_path}")

            self.sequential_training_step()

        print(f"\nTraining complete. Best reward: {best_score:.4f}")
        plot_rewards(rewards)
