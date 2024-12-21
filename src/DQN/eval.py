from magent2.environments import battle_v4
from src.DQN.torch_model import QNetwork
from src.DQN.final_torch_model import QNetwork as FinalQNetwork
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
    n_agent_each_team = len(env.env.action_spaces) // 2

    # Load models
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
        blue_wins, red_wins = [], []
        blue_rewards, red_rewards = [], []
        
        for _ in tqdm(range(n_episode)):
            env.reset()
            n_kills = {"blue": 0, "red": 0}
            episode_rewards = {"blue": 0, "red": 0}
            done = {agent: False for agent in env.agents}

            while not all(done.values()):
                for agent in env.agent_iter():
                    obs, reward, termination, truncation, _ = env.last()
                    agent_team = agent.split("_")[0]
                    
                    # Track kills and rewards
                    if reward > 4.5:  # Kill reward threshold
                        n_kills[agent_team] += 1
                    episode_rewards[agent_team] += reward

                    if termination or truncation:
                        action = None
                        done[agent] = True
                    else:
                        if agent_team == "blue":
                            action = blue_policy(env, agent, obs)
                        else:
                            action = red_policy(env, agent, obs)
                    env.step(action)

            # Determine winner based on kills
            who_wins = "red" if n_kills["red"] >= n_kills["blue"] + 5 else "draw"
            who_wins = "blue" if n_kills["blue"] >= n_kills["red"] + 5 else who_wins
            
            blue_wins.append(who_wins == "blue")
            red_wins.append(who_wins == "red")
            
            # Average rewards per agent
            blue_rewards.append(episode_rewards["blue"] / n_agent_each_team)
            red_rewards.append(episode_rewards["red"] / n_agent_each_team)

        return {
            "blue_winrate": np.mean(blue_wins),
            "red_winrate": np.mean(red_wins),
            "draw_rate": 1 - np.mean(blue_wins) - np.mean(red_wins),
            "blue_avg_reward": np.mean(blue_rewards),
            "red_avg_reward": np.mean(red_rewards)
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