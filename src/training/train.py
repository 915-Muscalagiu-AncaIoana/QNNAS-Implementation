import argparse
import traceback

from controller_trainer import DQNAgent
from data.dataset import Dataset, load_dataset_by_name
from domain.domain import TrainingStatus
from environment import QuantCircuitEnv
from repositories.training_session_repo import get_training_session_repository


def train_from_args(session_id, dataset_name, gates, discount, learning_rate, max_length, encoder_path=None):
    data_raw = load_dataset_by_name(dataset_name)
    dataset = Dataset(data_raw, encoder_path=encoder_path)

    # Create session if not provided
    if session_id is None:
        with get_training_session_repository() as repo:
            session = repo.create(
                dataset=dataset_name,
                gates=gates,
                discount_rate=discount,
                learning_rate=learning_rate,
                max_architecture_length=max_length,
                autoencoder_path=encoder_path,
            )
            session_id = session.id
            print(f"[INFO] Created new session with ID: {session_id}")

    print(f" Starting training for session {session_id}...")
    print(f" Dataset: {dataset_name}")
    print(f" Gates: {gates}")
    print(f" Discount rate (γ): {discount}")
    print(f" Learning rate: {learning_rate}")
    print(f" Max architecture length: {max_length}")
    if encoder_path:
        print(f" Using encoder at: {encoder_path}")

    try:
        with get_training_session_repository() as repo:
            repo.update_status(session_id, TrainingStatus.running)

        env = QuantCircuitEnv(max_length, dataset, gates, session_id)
        agent = DQNAgent(
            env=env,
            state_dim=env.max_length,
            action_dim=env.action_space.n,
            discount_rate=discount,
            learning_rate=learning_rate,
            session_id=session_id
        )
        agent.learn()

        with get_training_session_repository() as repo:
            repo.update_status(session_id, TrainingStatus.completed)

    except Exception as e:
        print(f"[ERROR] Training failed for session {session_id}: {e}")
        traceback.print_exc()

        with get_training_session_repository() as repo:
            repo.update_status(session_id, TrainingStatus.failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Quantum RL Agent")
    parser.add_argument("--session_id", type=int, required=False, help="Session ID from the database (optional)")
    parser.add_argument("--dataset", type=str, required=True, choices=["Iris", "Digits"], help="Dataset to use")
    parser.add_argument("--gates", nargs="+", required=True, help="Gates to allow (e.g., rx ry cz)")
    parser.add_argument("--discount", type=float, required=True, help="Discount factor γ")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=4, help="Max length of circuit architecture")
    parser.add_argument("--encoder_path", type=str, default=None, help="Path to encoder model (optional)")

    args = parser.parse_args()

    train_from_args(
        session_id=args.session_id,
        dataset_name=args.dataset,
        gates=args.gates,
        discount=args.discount,
        learning_rate=args.lr,
        max_length=args.max_length,
        encoder_path=args.encoder_path
    )
