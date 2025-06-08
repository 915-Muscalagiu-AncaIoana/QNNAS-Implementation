import torch
from training.controller_trainer import DQNAgent
from training.replay_memory import ReplayMemory


class DummyEnv:
    def __init__(self):
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.state_length = 4
        self.run_id = "dummy_run"

    def step(self, action):
        return [0.1, 0.2, 0.3, 0.4], 1.0, False, {}

    def reset(self, episode=None):
        return [0.0, 0.0, 0.0, 0.0]

    def render(self):
        pass


def test_policy_exploration(monkeypatch):
    agent = DQNAgent(env=None, state_dim=4, action_dim=3)

    monkeypatch.setattr("numpy.random.rand", lambda: 0.1)
    action = agent.epsilon_greedy_policy([0.5, 0.2, 0.1, 0.4], epsilon=0.9)
    assert 0 <= action < agent.action_dim


def test_policy_exploitation(monkeypatch):
    agent = DQNAgent(env=None, state_dim=4, action_dim=3)
    monkeypatch.setattr("numpy.random.rand", lambda: 0.99)
    monkeypatch.setattr(agent.model, "forward", lambda x: torch.tensor([[0.1, 0.9, 0.3]]))

    action = agent.epsilon_greedy_policy([0.5, 0.2, 0.1, 0.4], epsilon=0.1)
    assert action == 1


def test_play_one_step():
    env = DummyEnv()
    agent = DQNAgent(env=env, state_dim=4, action_dim=2)
    state = [0.0, 0.0, 0.0, 0.0]

    next_state, reward, done, info = agent.play_one_step(env, state, epsilon=1.0)
    assert len(agent.replay_memory.memory) == 1
    assert isinstance(next_state, list)
    assert reward == 1.0


def test_sequential_training_step():
    agent = DQNAgent(env=None, state_dim=4, action_dim=2)

    for _ in range(3):
        state = torch.rand(4, requires_grad=True)
        next_state = torch.rand(4)
        reward = 1.0
        action = 1
        done = False

        agent.replay_memory.memory.append((
            state.tolist(),
            action,
            reward,
            next_state.tolist(),
            done
        ))

    agent.sequential_training_step()

    # Check gradients computed for at least one parameter
    grads = [p.grad for p in agent.model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients were computed"
    assert any(torch.any(g != 0) for g in grads), "All gradients are zero"



def test_model_save_on_best(monkeypatch, tmp_path):
    env = DummyEnv()
    agent = DQNAgent(env, state_dim=4, action_dim=2, session_id="test123")

    monkeypatch.setattr(agent, "play_one_step", lambda e, s, eps: ([0, 0, 0, 0], 1.0, True, {}))
    monkeypatch.setattr(agent, "sequential_training_step", lambda: None)

    monkeypatch.setattr("training.controller_trainer.Path.cwd", lambda: tmp_path)
    monkeypatch.setattr("training.controller_trainer.plot_rewards", lambda r: None)
    monkeypatch.setattr("torch.save", lambda *args, **kwargs: print("torch.save called"))

    agent.learn()


def test_replay_memory_sample():
    memory = ReplayMemory(10)
    for i in range(5):
        memory.memory.append((i, i, i, i, i))
    sample = memory.sample(3)
    assert len(sample) == 3
