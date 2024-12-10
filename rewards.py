import torch
import numpy as np

def _calc_reward(rewards, state, lambda_reward=1.0):
    """
    Tính toán rewards dựa trên Lanchester strategy
    state shape: [batch, seq, H, W, channels]
    rewards shape: [batch, seq, n_agents, 1]
    """
    # Lấy coordinates của red và blue agents từ state
    red_coordinates = state[:, :, :, :, 1]  # [batch, seq, H, W]
    blue_coordinates = state[:, :, :, :, 3]  # [batch, seq, H, W]

    # Tính rewards cho từng agent
    batch_size = rewards.shape[0]
    seq_len = rewards.shape[1]
    
    env_rewards = rewards.clone()
    strategy_rewards = torch.zeros_like(rewards)
    
    for b in range(batch_size):
        for s in range(seq_len):
            # Lấy vị trí của tất cả blue agents trong timestep hiện tại
            blue_positions = blue_coordinates[b, s].nonzero()
            
            # Với mỗi agent (có thể ít hơn n_agents nếu một số đã chết)
            for idx, pos in enumerate(blue_positions):
                y, x = pos[0].item(), pos[1].item()
                
                # Kiểm tra vùng 3x3 cho consolidation
                y_start = max(0, y-1)
                y_end = min(45, y+2)
                x_start = max(0, x-1)
                x_end = min(45, x+2)
                
                # Kiểm tra vùng 5x5 cho overcrowding
                y_start_5x5 = max(0, y-2)
                y_end_5x5 = min(45, y+3)
                x_start_5x5 = max(0, x-2)
                x_end_5x5 = min(45, x+3)
                
                blue_force_3x3 = blue_coordinates[b, s, y_start:y_end, x_start:x_end].sum().cpu()
                red_force_3x3 = red_coordinates[b, s, y_start:y_end, x_start:x_end].sum().cpu()
                
                blue_force_5x5 = blue_coordinates[b, s, y_start_5x5:y_end_5x5, x_start_5x5:x_end_5x5].sum().cpu()

                if red_force_3x3 > 0: # Khi có địch
                    if blue_force_3x3 > red_force_3x3:
                        # Reward khi blue force lớn hơn
                        consolidation_reward = 2.0 / np.pi * np.arctan(blue_force_3x3 / red_force_3x3)
                        strategy_rewards[b, s, idx] += consolidation_reward
                    else:
                        # Penalty khi blue force nhỏ hơn, tỷ lệ với mức độ yếu thế
                        consolidation_penalty = -0.2 * (red_force_3x3 / blue_force_3x3)
                        strategy_rewards[b, s, idx] += consolidation_penalty
                elif blue_force_3x3 > 4: # Không có địch, tập trung lực lượng
                    # Thưởng thêm khi tập trung 5+ blue agents, nhưng có giới hạn
                    strategy_rewards[b, s, idx] += min(0.5, 0.1 * blue_force_3x3)
                else:
                    # Penalty giảm dần khi càng gần đủ 5 agents
                    strategy_rewards[b, s, idx] -= 0.1 * (5 - blue_force_3x3) / 5
                    
                # Phạt khi có quá nhiều agent trong vùng 5x5
                if blue_force_5x5 > 9:
                    overcrowding_penalty = -0.2 * (blue_force_5x5 - 9)
                    strategy_rewards[b, s, idx] += overcrowding_penalty

    # Normalize rewards
    strategy_rewards = (strategy_rewards - strategy_rewards.mean()) / (strategy_rewards.std() + 1e-8)
    
    # Tính mean reward cho các agent còn sống
    alive_mask = (env_rewards != 0).float()
    num_alive = alive_mask.sum(dim=2, keepdim=True)
    num_alive = torch.clamp(num_alive, min=1.0)
    env_rewards = (env_rewards * alive_mask).sum(dim=2, keepdim=True) / num_alive
    strategy_rewards = (strategy_rewards * alive_mask).sum(dim=2, keepdim=True) / num_alive
    final_rewards = lambda_reward * env_rewards + (1 - lambda_reward) * strategy_rewards
    
    return final_rewards.squeeze(-1)  # [batch, sequence, 1]

if __name__ == "__main__":
    rewards = torch.rand(1, 1000, 81, 1)
    state = torch.randint(0, 2, (1, 1000, 45, 45, 5))
    print(_calc_reward(rewards=rewards, state=state).shape)