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
    
    red_forces = get_cell_forces_vectorized(red_coordinates)
    blue_forces = get_cell_forces_vectorized(blue_coordinates)

    # Tính rewards cho từng agent
    batch_size = rewards.shape[0]
    seq_len = rewards.shape[1]
    n_agents = rewards.shape[2]
    
    env_rewards = rewards.clone()
    strategy_rewards = torch.zeros_like(rewards)
    
    for b in range(batch_size):
        for s in range(seq_len):
            # Lấy vị trí của tất cả blue agents trong timestep hiện tại
            blue_positions = blue_coordinates[b, s].nonzero()
            
            # Với mỗi agent (có thể ít hơn n_agents nếu một số đã chết)
            for idx, pos in enumerate(blue_positions):
                y, x = pos[0].item(), pos[1].item()
                
                # Tính consolidation of force reward
                # Tính số lượng đồng minh trong vùng 3x3
                y_start = max(0, y-1)
                y_end = min(45, y+2)
                x_start = max(0, x-1)
                x_end = min(45, x+2)
                
                blue_force = blue_coordinates[b, s, y_start:y_end, x_start:x_end].sum()
                red_force = red_coordinates[b, s, y_start:y_end, x_start:x_end].sum()
                
                if red_force > 0: # Khi có địch
                    if blue_force > red_force:
                        # Reward khi blue force lớn hơn
                        consolidation_reward = 2.0 / np.pi * np.arctan(blue_force / red_force)
                        strategy_rewards[b, s, idx] += consolidation_reward
                    else:
                        # Penalty khi blue force nhỏ hơn, tỷ lệ với mức độ yếu thế
                        consolidation_penalty = -0.2 * (red_force / blue_force) if blue_force > 0 else -0.3
                        strategy_rewards[b, s, idx] += consolidation_penalty
                elif blue_force > 4: # Không có địch, tập trung lực lượng
                    # Thưởng thêm khi tập trung 5+ blue agents, nhưng có giới hạn
                    strategy_rewards[b, s, idx] += min(0.5, 0.1 * blue_force)
                else:
                    # Penalty giảm dần khi càng gần đủ 5 agents
                    strategy_rewards[b, s, idx] -= 0.1 * (5 - blue_force) / 5

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

def get_cell_forces_vectorized(coordinates):
    """
    Computes cell forces in a vectorized manner using PyTorch, designed to work on GPU.
    
    Parameters:
        coordinates (torch.Tensor): A binary tensor of shape (batch, sequence, height, width).
                                    Non-zero values indicate the presence of agents.
    
    Returns:
        list: A list of dictionaries, where each dictionary contains the cell forces
              for a specific batch and sequence.
    """   
    # Find non-zero indices
    batch_indices, sequence_indices, height_indices, width_indices = coordinates.nonzero(as_tuple=True)
    
    # Combine height and width into tuples representing cells
    positions = torch.stack([height_indices, width_indices], dim=1)

    # Group by batch and sequence
    forces = []
    for b in range(coordinates.shape[0]):  # iterate over batch
        for s in range(coordinates.shape[1]):  # iterate over sequence
            # Filter cells belonging to the current batch and sequence
            mask = (batch_indices == b) & (sequence_indices == s)
            filtered_pos = positions[mask].tolist()
            
            # Convert cells to a dictionary with count 1 for each unique cell
            cell_forces = {tuple(cell): 1 for cell in filtered_pos}
            forces.append(cell_forces)

    return forces

if __name__ == "__main__":
    rewards = torch.rand(1, 1000, 81, 1)
    state = torch.randint(0, 2, (1, 1000, 45, 45, 5))
    print(_calc_reward(rewards=rewards, state=state).shape)