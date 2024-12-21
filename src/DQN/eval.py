import torch
from magent2.environments import battle_v4
import numpy as np
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Callable, Any, List, Tuple
from torch_model import QNetwork
from final_torch_model import QNetwork as FinalQNetwork

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x

@dataclass
class Config:
    max_cycles: int = 300
    map_size: int = 45
    n_episodes: int = 30
    kill_threshold: float = 4.5
    win_threshold: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    weights_dir: Path = Path("weight_models")

class ModelLoader:
    def __init__(self, config: Config):
        self.config = config
        
    def load_model(self, network: torch.nn.Module, model_name: str) -> torch.nn.Module:
        model_path = self.config.weights_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        network.load_state_dict(torch.load(str(model_path), map_location=self.config.device))
        network.eval()
        return network

class PolicyMaker:
    def __init__(self, config: Config):
        self.config = config
        
    def random_policy(self, env: Any, agent: str, obs: np.ndarray) -> int:
        return env.action_space(agent).sample()
    
    def network_policy(self, network: torch.nn.Module, env: Any, agent: str, obs: np.ndarray) -> int:
        observation = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            q_values = network(observation)
        return q_values.argmax().item()

class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.env = battle_v4.env(map_size=config.map_size, max_cycles=config.max_cycles)
        self.n_agent_each_team = len(self.env.env.action_spaces) // 2
        self.model_loader = ModelLoader(config)
        self.policy_maker = PolicyMaker(config)
        
    def run_eval(self, blue_policy: Callable, red_policy: Callable) -> Dict[str, float]:
        blue_wins, red_wins = [], []
        blue_rewards, red_rewards = [], []
        
        for _ in tqdm(range(self.config.n_episodes)):
            stats = self._run_episode(blue_policy, red_policy)
            blue_wins.append(stats["blue_win"])
            red_wins.append(stats["red_win"])
            blue_rewards.append(stats["blue_reward"])
            red_rewards.append(stats["red_reward"])
            
        return {
            "blue_winrate": np.mean(blue_wins),
            "red_winrate": np.mean(red_wins),
            "draw_rate": 1 - np.mean(blue_wins) - np.mean(red_wins),
            "blue_avg_reward": np.mean(blue_rewards),
            "red_avg_reward": np.mean(red_rewards)
        }
    
    def _run_episode(self, blue_policy: Callable, red_policy: Callable) -> Dict[str, Any]:
        self.env.reset()
        n_kills = {"blue": 0, "red": 0}
        episode_rewards = {"blue": 0, "red": 0}
        done = {agent: False for agent in self.env.agents}

        while not all(done.values()):
            for agent in self.env.agent_iter():
                obs, reward, termination, truncation, _ = self.env.last()
                agent_team = agent.split("_")[0]
                
                if reward > self.config.kill_threshold:
                    n_kills[agent_team] += 1
                episode_rewards[agent_team] += reward

                if termination or truncation:
                    action = None
                    done[agent] = True
                else:
                    action = blue_policy(self.env, agent, obs) if agent_team == "blue" else red_policy(self.env, agent, obs)
                self.env.step(action)

        who_wins = "red" if n_kills["red"] >= n_kills["blue"] + self.config.win_threshold else "draw"
        who_wins = "blue" if n_kills["blue"] >= n_kills["red"] + self.config.win_threshold else who_wins
        
        return {
            "blue_win": who_wins == "blue",
            "red_win": who_wins == "red",
            "blue_reward": episode_rewards["blue"] / self.n_agent_each_team,
            "red_reward": episode_rewards["red"] / self.n_agent_each_team
        }

    def evaluate_all(self):
        try:
            # Load networks
            blue_network = self.model_loader.load_model(
                QNetwork(self.env.observation_space("blue_0").shape, self.env.action_space("blue_0").n).to(self.config.device),
                "blue.pt"
            )
            red_network = self.model_loader.load_model(
                QNetwork(self.env.observation_space("red_0").shape, self.env.action_space("red_0").n).to(self.config.device),
                "red.pt"
            )
            red_final_network = self.model_loader.load_model(
                FinalQNetwork(self.env.observation_space("red_0").shape, self.env.action_space("red_0").n).to(self.config.device),
                "red_final.pt"
            )

            # Create policy functions
            blue_policy = lambda env, agent, obs: self.policy_maker.network_policy(blue_network, env, agent, obs)
            red_policy = lambda env, agent, obs: self.policy_maker.network_policy(red_network, env, agent, obs)
            red_final_policy = lambda env, agent, obs: self.policy_maker.network_policy(red_final_network, env, agent, obs)

            # Run evaluations
            scenarios = [
                ("blue.pt vs Random", self.policy_maker.random_policy),
                ("blue.pt vs red.pt", red_policy),
                ("blue.pt vs red_final.pt", red_final_policy)
            ]

            for name, red_policy in scenarios:
                print("=" * 50)
                print(f"Evaluating {name}")
                results = self.run_eval(blue_policy, red_policy)
                print("Results:", results)

        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise
        finally:
            self.env.close()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    config = Config()
    evaluator = Evaluator(config)
    evaluator.evaluate_all()

if __name__ == "__main__":
    main()