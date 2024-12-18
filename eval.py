from magent2.environments import battle_v4
from torch_model import QNetwork
from final_torch_model import QNetwork as FinalQNetwork
import torch
import numpy as np
import argparse

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # Fallback: tqdm becomes a no-op

class RuleBaseAgent:
    def __init__(self):
        """
        Khởi tạo RuleBaseAgent để xử lý observation batch
        """
        self.move_actions = {
            # (-2, 0): 0,   # Up 2
            (-1, -1): 1,  # Up Left
            (-1, 0): 2,   # Up 1
            (-1, 1): 3,   # Up Right
            # (0, -2): 4,   # Left 2
            (0, -1): 5,   # Left 1
            (0, 0): 6,    # Stay
            (0, 1): 7,    # Right 1
            # (0, 2): 8,    # Right 2
            (1, -1): 9,   # Down Left
            (1, 0): 10,   # Down 1
            (1, 1): 11,   # Down Right
            # (2, 0): 12    # Down 2
        }
        
        self.attack_actions = {
            (-1, -1): 13,  # Attack Up Left
            (-1, 0): 14,   # Attack Up
            (-1, 1): 15,   # Attack Up Right
            (0, -1): 16,   # Attack Left
            (0, 1): 17,    # Attack Right
            (1, -1): 18,   # Attack Down Left
            (1, 0): 19,    # Attack Down
            (1, 1): 20,    # Attack Down Right
        }

    def _get_agent_position(self):
        """
        Lấy vị trí của agent từ red_presence map
        """
        return torch.tensor([6, 6], dtype=torch.float32)

    def _direction_to_action(self, agent_pos, target_pos, attack=False):
        """
        Chuyển đổi hướng thành action, có xử lý trường hợp target nằm ngoài tầm
        """
        dy = target_pos[0] - agent_pos[0]
        dx = target_pos[1] - agent_pos[1]
        
        # Nếu là attack action và target trong tầm tấn công
        if attack and max(torch.abs(dy), torch.abs(dx)) <= 1:
            if (dy.item(), dx.item()) in self.attack_actions:
                return self.attack_actions[(dy.item(), dx.item())]
            return 6  # Nếu không có hướng tấn công phù hợp
            
        # Xử lý movement action
        if not attack:
            # Chuẩn hóa dy, dx để nằm trong phạm vi di chuyển
            dy = torch.clamp(dy, min=-2, max=2)
            dx = torch.clamp(dx, min=-2, max=2)
                
            # Kiểm tra xem có action tương ứng không
            if (dy.item(), dx.item()) in self.move_actions:
                return self.move_actions[(dy.item(), dx.item())]
            
            # Nếu không có action trực tiếp, tìm action di chuyển gần nhất
            min_dist = float('inf')
            best_action = 6
            
            for (move_dy, move_dx), action in self.move_actions.items():
                if action == 6:  # Bỏ qua action đứng yên
                    continue
                    
                # Tính khoảng cách sau khi thực hiện action này
                new_dy = dy - move_dy
                new_dx = dx - move_dx
                dist = torch.abs(new_dy) + torch.abs(new_dx)
                
                if dist < min_dist:
                    min_dist = dist
                    best_action = action
                    
            return best_action
            
        return 6  # Default: đứng yên

    def _logic(self, agent_id, red_presence, blue_presence):
        """
        Logic để chọn hành động dựa trên vị trí của đồng minh và địch
        """
        agent_pos = self._get_agent_position()
        
        # Tìm vị trí của địch và đồng minh
        enemy_positions = torch.nonzero(blue_presence > 0)
        ally_positions = torch.nonzero(red_presence > 0)
        
        # Loại bỏ vị trí của agent hiện tại khỏi ally_positions
        mask = ~torch.all(ally_positions == agent_pos, dim=1)
        ally_positions = ally_positions[mask]

        if len(enemy_positions) == 0:
            return torch.randint(0, 13, (1,)).item()  # Random movement nếu không có địch

        # Tính khoảng cách tới địch và đồng minh
        distances_to_enemies = torch.norm(enemy_positions - agent_pos, dim=1)
        closest_enemy_idx = torch.argmin(distances_to_enemies)
        closest_enemy_pos = enemy_positions[closest_enemy_idx]
        min_enemy_dist = distances_to_enemies[closest_enemy_idx]

        # Nếu có địch trong tầm tấn công (khoảng cách 1-sqrt(2))
        if min_enemy_dist < 2:
            return self._direction_to_action(agent_pos, closest_enemy_pos, attack=True)

        # Xác định target position dựa trên số lượng địch và đồng minh
        if len(enemy_positions) > len(ally_positions) and len(ally_positions) > 0:
            # Nếu địch đông hơn, di chuyển về phía đồng minh gần nhất
            distances_to_allies = torch.norm(ally_positions - agent_pos, dim=1)
            closest_ally_idx = torch.argmin(distances_to_allies)
            target_pos = ally_positions[closest_ally_idx]
        else:
            # Nếu địch ít hơn hoặc bằng, tấn công địch gần nhất
            target_pos = closest_enemy_pos

        return self._direction_to_action(agent_pos, target_pos)

    def get_action(self, obs_batch):
        """
        Xử lý batch observation và trả về actions cho tất cả agents
        """
        num_agents = obs_batch.shape[0]
        actions = []

        for agent_id in range(num_agents):
            obs = obs_batch[agent_id]  # [channels, height, width]
            red_presence = obs[1]  # Kênh 1: vị trí quân đỏ
            blue_presence = obs[3]  # Kênh 3: vị trí quân xanh
            action = self._logic(agent_id, red_presence, blue_presence)
            actions.append(action)

        return torch.tensor(actions)

def eval(args):
    max_cycles = 300
    env = battle_v4.env(map_size=45, max_cycles=max_cycles)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def random_policy(env, agent, obs):
        return env.action_space(agent).sample()

    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("red.pt", weights_only=True, map_location="cpu")
    )
    q_network.to(device)

    final_q_network = FinalQNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    final_q_network.load_state_dict(
        torch.load("red_final.pt", weights_only=True, map_location="cpu")
    )
    final_q_network.to(device)
    
    # Load blue agent
    blue_policy = RuleBaseAgent()

    def pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            q_values = q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]

    def final_pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            q_values = final_q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]

    def run_eval(env, red_policy, blue_policy, n_episode: int = 100):
        red_win, blue_win = [], []
        red_tot_rw, blue_tot_rw = [], []
        n_agent_each_team = len(env.env.action_spaces) // 2

        for _ in tqdm(range(n_episode)):
            env.reset()
            n_kill = {"red": 0, "blue": 0}
            red_reward, blue_reward = 0, 0

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                agent_team = agent.split("_")[0]

                n_kill[agent_team] += (
                    reward > 4.5
                )  # This assumes default reward settups
                if agent_team == "red":
                    red_reward += reward
                else:
                    blue_reward += reward

                if termination or truncation:
                    action = None  # this agent has died
                else:
                    if agent_team == "red":
                        action = red_policy(env, agent, observation)
                    else:
                        action = blue_policy.get_action(torch.from_numpy(observation).permute(2, 0, 1).unsqueeze(0))[0].item()

                env.step(action)

            who_wins = "red" if n_kill["red"] >= n_kill["blue"] + 5 else "draw"
            who_wins = "blue" if n_kill["red"] + 5 <= n_kill["blue"] else who_wins
            red_win.append(who_wins == "red")
            blue_win.append(who_wins == "blue")

            red_tot_rw.append(red_reward / n_agent_each_team)
            blue_tot_rw.append(blue_reward / n_agent_each_team)

        return {
            "winrate_red": np.mean(red_win),
            "winrate_blue": np.mean(blue_win),
            "average_rewards_red": np.mean(red_tot_rw),
            "average_rewards_blue": np.mean(blue_tot_rw),
        }

    print("=" * 20)
    print("Eval with random policy")
    print(
        run_eval(
            env=env, red_policy=random_policy, blue_policy=blue_policy, n_episode=30
        )
    )
    print("=" * 20)

    print("Eval with trained policy")
    print(
        run_eval(
            env=env, red_policy=pretrain_policy, blue_policy=blue_policy, n_episode=30
        )
    )
    print("=" * 20)

    print("Eval with final trained policy")
    print(
        run_eval(
            env=env,
            red_policy=final_pretrain_policy,
            blue_policy=blue_policy,
            n_episode=30,
        )
    )
    print("=" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate QMIX agents')
    parser.add_argument('--model_path', type=str, default='model/qmix', help='path to model')
    
    args = parser.parse_args()
    eval(args)
