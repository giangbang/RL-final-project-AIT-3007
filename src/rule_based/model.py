import torch

class RuleBasedAgent:
    def __init__(self, my_team):
        """
        Khởi tạo RuleBaseAgent để xử lý observation batch
        """
        self.my_team = my_team
        
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
            # Tính khoảng cách theo trục x đến các đồng minh
            x_distances = ally_positions[:, 1] - agent_pos[1]
            # Lọc ra các đồng minh bên trái (x_distance < 0) với Blue/ phải với Red
            allies_mask = x_distances < 0 if self.my_team == "blue" else x_distances > 0
            # 50% di chuyển về phía đồng minh xa nhất, 50% di chuyển ngẫu nhiên theo 4 hướng
            if torch.any(allies_mask):
                allies = ally_positions[allies_mask]
                # Trong số các đồng minh bên trái/phải, chọn đồng minh xa nhất
                distances_to_allies = torch.norm(allies - agent_pos, dim=1)
                farthest_ally_idx = torch.argmax(distances_to_allies)
                target_pos = allies[farthest_ally_idx]
                return self._direction_to_action(agent_pos, target_pos) if torch.rand(1).item() < 0.5 else torch.randint(0, 4, (1,)).item() * 4
            else:
                # Nếu không có đồng minh bên trái/phải, di chuyển ngẫu nhiên 2 bước theo 4 hướng
                return torch.randint(0, 4, (1,)).item() * 4 # Random giữa Up 2, Left 2, Right 2, Down 2

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
        Input: obs_batch - [batch*n_agent, channels, height, width]
        Output: actions - [batch*n_agent]
        Khi evaluate batch=1, n_agent=1: 
        actions - [1]
        action = actions[0]
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
    
# rule_base_agent = RuleBasedAgent(my_team='red')