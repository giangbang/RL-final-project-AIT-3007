import torch

from magent2.environments import battle_v4
from src.qmix.qmix import CNNFeatureExtractor
from src.rnn_agent.rnn_agent import RNN_Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_blue_policy(model_path, hidden_dim=64):
    """
    Khởi tạo policy cho team blue sử dụng QMIX model đã train
    """
    # Khởi tạo environment để lấy các thông số
    dummy_env = battle_v4.env(map_size=45, minimap_mode=False, extra_features=False)
    dummy_env.reset()
    
    # Khởi tạo CNN feature extractor để lấy kích thước output
    dummy_cnn = CNNFeatureExtractor()
    obs_dim = dummy_cnn.get_output_dim(dummy_env.observation_space("blue_0").shape[:-1])
    state_dim = dummy_cnn.get_output_dim(dummy_env.state().shape[:-1])
    action_dim = dummy_env.action_space("blue_0").n
    action_shape = 1
    n_agents = len(dummy_env.agents)//2
    
    learner = RNN_Trainer(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_shape=action_shape,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        target_update_interval=10,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0
    )
    
    # Load model đã train
    learner.load_model(model_path, map_location=device)
    
    # Khởi tạo hidden state
    hidden_states = {i: torch.zeros(1, 1, hidden_dim).to(device) for i in range(n_agents)}
    
    def policy(env, agent_id, obs):
        """
        Policy cho một agent trong team blue
        """
        nonlocal hidden_states
        
        # Lưu observation vào dictionary
        agent_idx = int(agent_id.split("_")[1])

        # Get action từ model
        action, new_hidden = learner.get_action(obs, hidden_states[agent_idx])
        hidden_states[agent_idx] = new_hidden
        
        # Trả về action cho agent hiện tại
        return action[0][0]
        
    return policy