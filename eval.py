from magent2.environments import battle_v4
from torch_model import QNetwork
import torch
import numpy as np


def eval():
    # this function test with random policy
    env = battle_v4.env(map_size=45)

    def random_policy(env, agent, obs):
        return env.action_space(agent).sample()

    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("red.pt", weights_only=True, map_location="cpu")
    )

    def pretrain_policy(env, agent, obs):
        observation = torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(observation)
        return torch.argmax(q_values, dim=1).numpy()[0]

    def run_eval(env, red_policy, blue_policy, n_episode: int = 100):
        red_win, blue_win = [], []
        red_tot_rw, blue_tot_rw = [], []

        for _ in range(n_episode):
            env.reset()
            red_alive, blue_alive = {}, {}
            red_reward, blue_reward = 0, 0

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                agent_team = agent.split("_")[0]
                if agent_team == "red":
                    red_reward += reward
                    red_alive[agent] = ~termination  # truncation is for timeout
                else:
                    blue_reward += reward
                    blue_alive[agent] = ~termination  # truncation is for timeout

                if termination or truncation:
                    action = None  # this agent has died
                else:
                    if agent_team == "red":
                        action = red_policy(env, agent, observation)
                    else:
                        action = blue_policy(env, agent, observation)

                env.step(action)

            n_blue_agent_alive = np.sum(np.fromiter(blue_alive.values(), dtype=bool))
            n_red_agent_alive = np.sum(np.fromiter(red_alive.values(), dtype=bool))

            red_win.append(n_red_agent_alive > n_blue_agent_alive)
            blue_win.append(n_blue_agent_alive > n_red_agent_alive)

            red_tot_rw.append(red_reward / len(red_alive))
            blue_tot_rw.append(blue_reward / len(blue_alive))

        return {
            "winrate_red": np.mean(red_win),
            "winrate_blue": np.mean(blue_win),
            "average_rewards_red": np.mean(red_tot_rw),
            "average_rewards_blue": np.mean(blue_tot_rw),
        }

    print("=" * 20)
    print("Eval with random policy")
    print(run_eval(env=env, red_policy=random_policy, blue_policy=random_policy))
    print("=" * 20)

    print("Eval with trained policy")
    print(run_eval(env=env, red_policy=pretrain_policy, blue_policy=random_policy))
    print("=" * 20)


if __name__ == "__main__":
    eval()
