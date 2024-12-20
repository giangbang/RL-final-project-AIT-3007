from train import train
import wandb
import argparse
from torchrl.data import TensorDictReplayBuffer, RandomSampler, LazyMemmapStorage
import tempfile

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="QMIX training for MAgent2 battle environment")

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=99,
                        help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Starting value of epsilon for epsilon-greedy exploration")
    parser.add_argument("--epsilon_decay", type=float, default=0.994,
                        help="Decay rate of epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.05,
                        help="Minimum value of epsilon")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of episodes to train")
    parser.add_argument("--update_step", type=int, default=50,
                        help="Number of steps between target network updates")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft update coefficient for target networks")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    # Initialize wandb
    wandb.login(key="37d305fc5fac9b15b88ec48208250be56185986e")  # Replace with your actual API key or handle securely
    wandb.init(
        project="QMIX_Project_RL_new",
        config={
            "seed": args.seed,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_decay": args.epsilon_decay,
            "epsilon_min": args.epsilon_min,
            "num_episodes": args.num_episodes,
            "update_step": args.update_step,
            "tau": args.tau,
        }
    )
    config = wandb.config

    # Start training
    train(config)