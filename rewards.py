import torch
import numpy as np

def _calc_reward(rewards, state, lambda_env=1.0, lambda_strategy=1.0):
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
                if idx < n_agents:  # Chỉ xét n_agents đầu tiên
                    cell = (pos[0].item(), pos[1].item())
                    
                    # Tính consolidation of force reward
                    red_force = red_forces[b*seq_len + s].get(cell, 0)
                    blue_force = blue_forces[b*seq_len + s].get(cell, 0)
                    
                    if blue_force > 0 and red_force > 0:
                        if blue_force > red_force:
                            # Reward khi blue force lớn hơn
                            consolidation_reward = 2.0 / np.pi * np.arctan(blue_force / red_force)
                            strategy_rewards[b, s, idx] += consolidation_reward
                        else:
                            # Penalty khi blue force nhỏ hơn
                            strategy_rewards[b, s, idx] -= 0.1
                    elif blue_force > 0 and red_force == 0:
                        # Thưởng thêm nếu cell chỉ có blue agents
                        strategy_rewards[b, s, idx] += 0.5

    # Normalize rewards
    strategy_rewards = (strategy_rewards - strategy_rewards.mean()) / (strategy_rewards.std() + 1e-8)
    
    # Tính mean reward cho các agent còn sống
    alive_mask = (env_rewards != 0).float()
    num_alive = alive_mask.sum(dim=2, keepdim=True)
    num_alive = torch.clamp(num_alive, min=1.0)
    env_rewards = (env_rewards * alive_mask).sum(dim=2, keepdim=True) / num_alive
    strategy_rewards = (strategy_rewards * alive_mask).sum(dim=2, keepdim=True) / num_alive
    final_rewards = lambda_env * env_rewards + lambda_strategy * strategy_rewards
    
    return final_rewards.squeeze(-1)  # [batch, sequence, 1]

def get_cell_forces(coordinates):
    """
    Tính force cho mỗi cell (tính tuần tự, đơn giản nhưng tốn thời gian)
    """
    forces = []
    for b in range(coordinates.shape[0]):  # batch
        for s in range(coordinates.shape[1]):  # sequence
            # Đếm số lượng agent trong mỗi cell
            cell_forces = {}
            positions = coordinates[b,s].nonzero()
            for pos in positions:
                cell = (pos[0].item(), pos[1].item())
                if cell not in cell_forces:
                    cell_forces[cell] = 1
                else:
                    cell_forces[cell] += 1
            forces.append(cell_forces)
    return forces

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
    cells = torch.stack([height_indices, width_indices], dim=1)

    # Group by batch and sequence
    forces = []
    for b in range(coordinates.shape[0]):  # iterate over batch
        for s in range(coordinates.shape[1]):  # iterate over sequence
            # Filter cells belonging to the current batch and sequence
            mask = (batch_indices == b) & (sequence_indices == s)
            filtered_cells = cells[mask].tolist()
            
            # Convert cells to a dictionary with count 1 for each unique cell
            cell_forces = {tuple(cell): 1 for cell in filtered_cells}
            forces.append(cell_forces)

    return forces

if __name__ == "__main__":
    rewards = torch.rand(1, 1000, 81, 1)
    state = torch.randint(0, 2, (1, 1000, 45, 45, 5))
    print(_calc_reward(rewards=rewards, state=state).shape)