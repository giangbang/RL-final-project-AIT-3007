import torch
from magent2.environments import battle_v4
import os
import cv2
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Any
from torch_model import QNetwork
from final_torch_model import QNetwork as FinalQNetwork

@dataclass
class Config:
    map_size: int = 45
    fps: int = 35
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    video_dir: str = "video"
    weights_dir: str = "weight_models"

class VideoRecorder:
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.video_dir, exist_ok=True)

    def create_video(self, frames: List[Any], filename: str) -> None:
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(
            os.path.join(self.config.video_dir, filename),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.config.fps,
            (width, height),
        )
        try:
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        finally:
            out.release()

class ModelLoader:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

    def load_model(self, network: torch.nn.Module, model_name: str) -> torch.nn.Module:
        model_path = Path(self.config.weights_dir) / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        network.load_state_dict(torch.load(str(model_path), map_location=self.device))
        network.eval()
        return network

class PolicyMaker:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

    def random_policy(self, obs: Any, env: Any, agent: str) -> int:
        return env.action_space(agent).sample()

    def network_policy(self, network: torch.nn.Module, obs: Any) -> int:
        observation = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = network(observation)
        return q_values.argmax().item()

class BattleSimulator:
    def __init__(self, config: Config):
        self.config = config
        self.env = battle_v4.env(map_size=config.map_size, render_mode="rgb_array")
        self.model_loader = ModelLoader(config)
        self.video_recorder = VideoRecorder(config)
        self.policy_maker = PolicyMaker(config)

    def run_episode(self, blue_network: torch.nn.Module, red_policy: Callable) -> List[Any]:
        frames = []
        self.env.reset()
        
        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()

            if termination or truncation:
                action = None
            else:
                agent_team = agent.split("_")[0]
                if agent_team == "blue":
                    action = self.policy_maker.network_policy(blue_network, observation)
                else:
                    action = red_policy(observation, self.env, agent)

            self.env.step(action)
            if agent == "blue_0":
                frames.append(self.env.render())
                
        return frames

    def run_simulation(self):
        try:
            # Initialize networks
            blue_network = self.model_loader.load_model(
                QNetwork(
                    self.env.observation_space("blue_0").shape,
                    self.env.action_space("blue_0").n
                ).to(self.config.device),
                "blue.pt"
            )

            red_network = self.model_loader.load_model(
                QNetwork(
                    self.env.observation_space("red_0").shape,
                    self.env.action_space("red_0").n
                ).to(self.config.device),
                "red.pt"
            )

            red_final_network = self.model_loader.load_model(
                FinalQNetwork(
                    self.env.observation_space("red_0").shape,
                    self.env.action_space("red_0").n
                ).to(self.config.device),
                "red_final.pt"
            )

            # Record episodes
            scenarios = [
                ("blue_vs_random.mp4", self.policy_maker.random_policy),
                ("blue_vs_red.mp4", lambda obs, env, agent: self.policy_maker.network_policy(red_network, obs)),
                ("blue_vs_red_final.mp4", lambda obs, env, agent: self.policy_maker.network_policy(red_final_network, obs))
            ]

            for filename, red_policy in scenarios:
                frames = self.run_episode(blue_network, red_policy)
                self.video_recorder.create_video(frames, filename)
                logging.info(f"Done recording {filename}")

        except Exception as e:
            logging.error(f"Error during simulation: {e}")
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
    simulator = BattleSimulator(config)
    simulator.run_simulation()

if __name__ == "__main__":
    main()