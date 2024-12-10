import torch
import numpy as np

def _calc_reward(rewards, state, action, lambda_reward=1.0):
    """
    Tính toán rewards dựa trên chiến thuật tổng hợp cho blue team
    state shape: [batch, seq, H, W, channels]
    rewards shape: [batch, seq, n_agents, 1]
    action shape: [batch, seq, n_agents, 1]
    """
    # Lấy coordinates của red và blue agents từ state
    red_coordinates = state[:, :, :, :, 1]  # [batch, seq, H, W]
    blue_coordinates = state[:, :, :, :, 3]  # [batch, seq, H, W]

    # Tính rewards cho từng agent
    batch_size = rewards.shape[0]
    seq_len = rewards.shape[1]
    
    env_rewards = rewards.clone().squeeze(-1)
    strategy_rewards = torch.zeros((batch_size, seq_len, 1)).to(rewards.device)
    
    for b in range(batch_size):
        for t in range(seq_len):
            # Đếm số lượng agents
            red_count = red_coordinates[b,t].sum()
            blue_count = blue_coordinates[b,t].sum()
            
            # Tính center of mass cho mỗi team
            red_pos = torch.nonzero(red_coordinates[b,t])
            blue_pos = torch.nonzero(blue_coordinates[b,t])
            
            if len(red_pos) > 0 and len(blue_pos) > 0:
                red_center = red_pos.float().mean(dim=0)
                blue_center = blue_pos.float().mean(dim=0)
                
                # 1. Cohesion reward: thưởng khi đội hình tập trung
                blue_spread = torch.pdist(blue_pos.float()).mean() if len(blue_pos) > 1 else torch.tensor(0.0)
                cohesion_reward = -blue_spread * 0.01
                
                # 2. Numerical advantage reward: thưởng khi có lợi thế về số lượng
                number_advantage = (blue_count - red_count) * 0.1
                
                # 3. Surrounding reward: thưởng khi bao vây đối phương
                angles = torch.atan2(blue_pos[:,0] - red_center[0], 
                                    blue_pos[:,1] - red_center[1])
                angle_diff = torch.sort(angles)[0]
                if len(angle_diff) > 1:
                    max_gap = torch.max(angle_diff[1:] - angle_diff[:-1])
                    surrounding_reward = (2*np.pi - max_gap) * 0.1
                else:
                    surrounding_reward = torch.tensor(0.0)
                
                # 4. Strategic positioning reward: thưởng khi giữ khoảng cách hợp lý
                dist_to_enemy = torch.norm(blue_center - red_center)
                position_reward = -torch.abs(dist_to_enemy - 10.0) * 0.05  # optimal distance = 10
                
                strategy_rewards[b,t] = (cohesion_reward + number_advantage + 
                                       surrounding_reward + position_reward)
    
    # Normalize strategy rewards
    if strategy_rewards.std() > 0:
        strategy_rewards = (strategy_rewards - strategy_rewards.mean()) / strategy_rewards.std()
    

    # Tính mean reward cho các agent còn sống
    alive_mask = (env_rewards != 0).float()
    num_alive = alive_mask.sum(dim=2, keepdim=True)
    num_alive = torch.clamp(num_alive, min=1.0)
    env_rewards = (env_rewards * alive_mask).sum(dim=2, keepdim=True) / num_alive
    final_rewards = lambda_reward * env_rewards + (1 - lambda_reward) * strategy_rewards
    
    return final_rewards.squeeze(-1)  # [batch, sequence, 1]

if __name__ == "__main__":
    rewards = torch.rand(1, 1000, 81, 1)
    state = torch.randint(0, 2, (1, 1000, 45, 45, 5))
    print(_calc_reward(rewards=rewards, state=state).shape)