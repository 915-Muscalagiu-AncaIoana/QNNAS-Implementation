from sklearn.datasets import load_iris, load_digits
import argparse
from controller_trainer import DQNAgent
from data.dataset import Dataset
from environment import QuantCircuitEnv
import uuid

def train_from_args(dataset_name, gates, discount, learning_rate, max_length, encoder_path=None):
    print(f" Starting training...")
    print(f" Dataset: {dataset_name}")
    print(f" Gates: {gates}")
    print(f" Discount rate (γ): {discount}")
    print(f" Learning rate: {learning_rate}")
    print(f" Max architecture length: {max_length}")
    if encoder_path:
        print(f" Using encoder at: {encoder_path}")

    run_id = uuid.uuid4()
    if dataset_name == 'Iris':
        data = load_iris()
    else:
        data = load_digits()

    dataset = Dataset(data, encoder_path=encoder_path)
    env = QuantCircuitEnv(max_length, dataset, gates, run_id)
    agent = DQNAgent(
        env=env,
        state_dim=env.max_length,
        action_dim=env.action_space.n,
        discount_rate=discount,
        learning_rate=learning_rate
    )
    agent.learn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Quantum RL Agent")
    parser.add_argument("--dataset", type=str, required=True, choices=["Iris", "Digits"], help="Dataset to use")
    parser.add_argument("--gates", nargs="+", required=True, help="Gates to allow (e.g., rx ry cz)")
    parser.add_argument("--discount", type=float, required=True, help="Discount factor γ")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=4, help="Max length of circuit architecture")
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Path to encoder model to compress input features (optional)")
    args = parser.parse_args()
    train_from_args(args.dataset, args.gates, args.discount, args.lr, args.max_length, encoder_path=args.encoder_path)
