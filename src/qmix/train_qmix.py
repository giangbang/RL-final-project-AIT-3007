import numpy as np
import torch
import argparse
import random
from magent2.environments import battle_v4
from src.qmix.qmix import QMix_Trainer, ReplayBuffer
from src.cnn import CNNFeatureExtractor
from src.qmix.utils import get_all_states, make_action
from torch_model import QNetwork
from src.rnn_agent.rnn_agent import RNN_Trainer, ReplayBufferGRU

# Thêm đoạn parse arguments trước khi định nghĩa các biến
parser = argparse.ArgumentParser(description='Train QMIX agents')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--max_episodes', type=int, default=640, help='maximum number of episodes')
parser.add_argument('--max_steps', type=int, default=1000, help='maximum steps per episode')
parser.add_argument('--save_interval', type=int, default=20, help='interval to save model')
parser.add_argument('--target_update_interval', type=int, default=10, help='interval to update target network')
parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon for exploration')
parser.add_argument('--epsilon_end', type=float, default=0.05, help='Minimum epsilon value')
parser.add_argument('--epsilon_decay', type=float, default=0.985, help='Epsilon decay rate')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--lambda_reward', type=float, default=0, help='Weight reward from enviroment')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
parser.add_argument('--model_path', type=str, default='model/qmix', help='Path to save model')
parser.add_argument('--red_pretrained', action='store_true', help='Use red.pt pretrained model')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')

args = parser.parse_args()

dummy_cnn = CNNFeatureExtractor()
replay_buffer_size = args.batch_size
hidden_dim = 64
hypernet_dim = 128
max_steps = args.max_steps
max_episodes = args.max_episodes
batch_size = args.batch_size
save_interval = args.save_interval
target_update_interval = args.target_update_interval
model_path = args.model_path
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
dead_penalty=-1, attack_penalty=-0.1, attack_opponent_reward=1.0,
max_cycles=300, extra_features=False)

env.reset()
obs_dim = dummy_cnn.get_output_dim(env.observation_space("blue_0").shape[:-1])
state_dim = dummy_cnn.get_output_dim(env.state().shape[:-1])
action_dim = env.action_space("blue_0").n
action_shape = 1
n_agents = len(env.agents)//2

# replay_buffer = ReplayBuffer(replay_buffer_size)
# learner = QMix_Trainer(
#     replay_buffer=replay_buffer,
#     n_agents=n_agents,
#     obs_dim=obs_dim,
#     state_dim=state_dim,
#     action_shape=action_shape,
#     action_dim=action_dim,
#     hidden_dim=hidden_dim,
#     hypernet_dim=hypernet_dim,
#     target_update_interval=target_update_interval,
#     lr=args.learning_rate,
#     epsilon_start=args.epsilon_start,
#     epsilon_end=args.epsilon_end,
#     epsilon_decay=args.epsilon_decay,
#     lambda_reward=args.lambda_reward,
# )

replay_buffer = ReplayBufferGRU(replay_buffer_size)
learner = RNN_Trainer(
    replay_buffer=replay_buffer,
    n_agents=n_agents,
    obs_dim=obs_dim,
    action_shape=action_shape,
    action_dim=action_dim,
    hidden_dim=hidden_dim,
    target_update_interval=target_update_interval,
    lr=args.learning_rate,
    epsilon_start=args.epsilon_start,
    epsilon_end=args.epsilon_end,
    epsilon_decay=args.epsilon_decay,
    lambda_reward=args.lambda_reward,
)

if args.checkpoint:
    learner.load_model(args.checkpoint, map_location=device)
    
red_agent = None
if args.red_pretrained:
    # Red.pt
    red_agent = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    red_agent.load_state_dict(
        torch.load("red.pt", weights_only=True, map_location="cpu")
    )
    red_agent.to(device)

def train_blue_qmix(env, learner, max_episodes=1000, max_steps=200, batch_size=32, 
                    save_interval=100, model_path='model/qmix'):
    """
    Train blue team using QMIX algorithm
    
    Args:
        env: MAgent environment
        learner: QMix_Trainer instance
        max_episodes: Maximum number of episodes
        max_steps: Maximum steps per episode
        batch_size: Batch size for training
        save_interval: Interval to save model
        model_path: Path to save model
    """
    learner.agent.train()
    learner.target_agent.train()
    # learner.mixer.train()
    # learner.target_mixer.train()
    loss, strategy_reward, env_reward, target_reward = None, None, None, None
    for episode in range(max_episodes):
        print(f"Start episode {episode} ----------------------")
        env.reset()
        episode_reward = 0
        
        # Clear memory after each episode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize hidden states for all blue agents
        hidden_out = torch.zeros(1, 1, n_agents, hidden_dim).to(device)
        # hidden_in = hidden_out.clone()
        
        # Lists to store episode data
        episode_observations = []
        # episode_states = []
        # episode_next_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_observations = []
        
        # Init dead agents list
        dead_agents = []
        
        for step in range(max_steps):
            hidden_in = hidden_out

            if all(env.truncations.values()):
                print(step)
                print("Environment truncated!!!")
                break
            # Get all blue agents states and rewards
            observations, rewards, terminations, truncations, infos = get_all_states(env, dead_agents)
            if len(observations) == 0:  # No blue agents alive
                break
            observations = np.stack(observations) # [n_agents, obs_dim]

            # Get actions from RNNAgent
            actions, hidden_out = learner.get_action(observations, hidden_in)

            # Execute actions and collect next states/rewards
            # Save dead agents after making actions
            dead_agents = make_action(actions, env, dead_agents, red_agent)
            next_observations, rewards, terminations, truncations, infos = get_all_states(env, dead_agents)
            if len(next_observations) == 0:  # No blue agents alive
                break
            next_observations = np.stack(next_observations) # [n_agents, obs_dim]
            rewards = np.stack(rewards) # [n_agents]

            if step == 0:   
                ini_hidden_in = hidden_in
                ini_hidden_out = hidden_out
            
            # Store transition
            episode_observations.append(observations)
            # episode_states.append(state)
            # episode_next_states.append(next_state)
            episode_actions.append(actions)
            episode_rewards.append(rewards)
            episode_next_observations.append(next_observations)
            
            episode_reward += rewards.sum()
            
        # print(np.stack(episode_states).shape) #(1000, 81, 845)
        # print(np.stack(episode_actions).shape) #(1000, 81, 1)
        # print(np.stack(episode_rewards).shape) #(1000, 81)
        # print(np.stack(episode_next_states).shape) #(1000, 81, 845)

        # Push entire episode to replay buffer
        # episode_states = np.stack(episode_states)
        # episode_next_states = np.stack(episode_next_states)
        # if len(episode_observations) > 0:
        #     learner.push_replay_buffer(
        #         ini_hidden_in=ini_hidden_in,
        #         ini_hidden_out=ini_hidden_out,
        #         episode_observation=episode_observations,
        #         episode_state=episode_states,
        #         episode_next_state=episode_next_states,
        #         episode_action=episode_actions,
        #         episode_reward=episode_rewards,
        #         episode_next_observation=episode_next_observations
        #     )
        
        if len(episode_observations) > 0:
            learner.push_replay_buffer(
                ini_hidden_in=ini_hidden_in,
                ini_hidden_out=ini_hidden_out,
                episode_observation=episode_observations,
                episode_action=episode_actions,
                episode_reward=episode_rewards,
                episode_next_observation=episode_next_observations
            )
        
        # Clear unnecessary tensors
        del hidden_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Training step
        if episode + 1 >= batch_size:
            loss, target_reward, env_reward, strategy_reward = learner.update(batch_size)

        # Save model periodically
        if episode % save_interval == 0:
            learner.save_model(f"{model_path}_episode_{episode}")
            
        print(f"Episode {episode}: Reward = {episode_reward/n_agents:.2f}, TR = {np.round(target_reward,2) if target_reward else 'N/A'}, ER = {np.round(env_reward,2) if env_reward else 'N/A'}, SR = {np.round(strategy_reward,2) if strategy_reward else 'N/A'}, Loss = {loss if loss else 'N/A'}")
    
    # Save final model
    learner.save_model(model_path)
    env.close()
    
    return learner

def set_seed(seed):
    print(seed)
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(args.seed)
    # Sử dụng hàm training
    trained_qmix = train_blue_qmix(
        env=env,
        learner=learner,
        max_episodes=max_episodes,
        max_steps=max_steps,
        batch_size=batch_size,
        save_interval=save_interval,
        model_path=model_path
    )