from magent2.environments import battle_v4
from torch_model import QNetwork
import torch
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # Fallback: tqdm becomes a no-op


def eval():
    max_cycles = 300
    env = battle_v4.env(map_size=45, max_cycles=max_cycles)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Random policy
    def random_policy(env, agent, obs):
        return env.action_space(agent).sample()

    # Load pretrained blue agent
    blue_q_network = QNetwork(
        env.observation_space("blue_0").shape, env.action_space("blue_0").n
    )
    blue_q_network.load_state_dict(torch.load("blue.pt", map_location=device))
    blue_q_network.to(device)
    blue_q_network.eval()

    def blue_pretrained_policy(env, agent, obs):
        observation = (
            torch.tensor(obs, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            action = blue_q_network(observation).argmax().item()
        return action

    # Load pretrained red agent
    red_q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    red_q_network.load_state_dict(torch.load("red.pt", map_location=device))
    red_q_network.to(device)
    red_q_network.eval()

    def red_pretrained_policy(env, agent, obs):
        observation = (
            torch.tensor(obs, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            action = red_q_network(observation).argmax().item()
        return action

    def run_eval(env, blue_policy, red_policy, n_episode=10):
        results = {"blue_win": 0, "red_win": 0, "draw": 0}
        for _ in tqdm(range(n_episode)):
            env.reset()
            dones = {agent: False for agent in env.agents}
            while not all(dones.values()):
                for agent in env.agent_iter():
                    obs, reward, termination, truncation, _ = env.last()
                    if termination or truncation:
                        action = None
                        dones[agent] = True
                    else:
                        if agent.startswith("blue"):
                            action = blue_policy(env, agent, obs)
                        else:
                            action = red_policy(env, agent, obs)
                    env.step(action)
            blue_alive = sum(
                1 for agent in env.agents if agent.startswith("blue")
            )
            red_alive = sum(
                1 for agent in env.agents if agent.startswith("red")
            )
            if blue_alive > red_alive:
                results["blue_win"] += 1
            elif red_alive > blue_alive:
                results["red_win"] += 1
            else:
                results["draw"] += 1
        return results

    print("=" * 20)
    print("Eval: Blue (trained) vs Red (random)")
    print(
        run_eval(
            env=env,
            blue_policy=blue_pretrained_policy,
            red_policy=random_policy,
            n_episode=30,
        )
    )
    print("=" * 20)

    print("Eval: Blue (trained) vs Red (trained)")
    print(
        run_eval(
            env=env,
            blue_policy=blue_pretrained_policy,
            red_policy=red_pretrained_policy,
            n_episode=30,
        )
    )
    print("=" * 20)


if __name__ == "__main__":
    eval()
