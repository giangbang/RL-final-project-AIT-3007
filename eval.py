from magent2.environments import battle_v4
from torch_model import QNetwork
from final_torch_model import QNetwork as FinalQNetwork
import torch
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x

def eval():
    max_cycles = 300
    env = battle_v4.env(map_size=45, max_cycles=max_cycles)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all models
    blue_network = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device)
    blue_network.load_state_dict(torch.load("blue.pt", map_location=device))
    blue_network.eval()

    red_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n).to(device)
    red_network.load_state_dict(torch.load("red.pt", map_location=device))
    red_network.eval()

    red_final_network = FinalQNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n).to(device)
    red_final_network.load_state_dict(torch.load("red_final.pt", map_location=device))
    red_final_network.eval()

    # Define policies
    def random_policy(env, agent, obs):
        return env.action_space(agent).sample()

    def blue_policy(env, agent, obs):
        observation = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = blue_network(observation)
        return q_values.argmax().item()

    def red_policy(env, agent, obs):
        observation = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = red_network(observation)
        return q_values.argmax().item()

    def red_final_policy(env, agent, obs):
        observation = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = red_final_network(observation)
        return q_values.argmax().item()

    def run_eval(env, blue_policy, red_policy, n_episode=30):
        blue_wins, red_wins, draws = 0, 0, 0
        blue_total_reward, red_total_reward = 0, 0

        for _ in tqdm(range(n_episode)):
            env.reset()
            episode_blue_reward, episode_red_reward = 0, 0
            done = {agent: False for agent in env.agents}

            while not all(done.values()):
                for agent in env.agent_iter():
                    obs, reward, termination, truncation, _ = env.last()
                    if termination or truncation:
                        action = None
                        done[agent] = True
                    else:
                        if agent.startswith("blue"):
                            action = blue_policy(env, agent, obs)
                            episode_blue_reward += reward
                        else:
                            action = red_policy(env, agent, obs)
                            episode_red_reward += reward
                    env.step(action)

            blue_alive = sum(1 for agent in env.agents if agent.startswith("blue"))
            red_alive = sum(1 for agent in env.agents if agent.startswith("red"))

            if blue_alive > red_alive:
                blue_wins += 1
            elif red_alive > blue_alive:
                red_wins += 1
            else:
                draws += 1

            blue_total_reward += episode_blue_reward
            red_total_reward += episode_red_reward

        return {
            "blue_wins": blue_wins,
            "red_wins": red_wins,
            "draws": draws,
            "win_rate_blue": blue_wins/n_episode,
            "win_rate_red": red_wins/n_episode,
            "avg_reward_blue": blue_total_reward/n_episode,
            "avg_reward_red": red_total_reward/n_episode
        }

    # Run evaluations
    print("=" * 50)
    print("Evaluating blue.pt vs Random")
    results = run_eval(env, blue_policy, random_policy)
    print("Results:", results)
    print("=" * 50)

    print("Evaluating blue.pt vs red.pt")
    results = run_eval(env, blue_policy, red_policy)
    print("Results:", results)
    print("=" * 50)

    print("Evaluating blue.pt vs red_final.pt")
    results = run_eval(env, blue_policy, red_final_policy)
    print("Results:", results)
    print("=" * 50)

if __name__ == "__main__":
    eval()