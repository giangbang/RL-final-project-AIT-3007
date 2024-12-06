import numpy as np
import torch
import argparse

from magent2.environments import battle_v4
from qmix import QMix_Trainer, ReplayBufferGRU
from utils import get_all_states, make_action

# Thêm đoạn parse arguments trước khi định nghĩa các biến
parser = argparse.ArgumentParser(description='Train QMIX agents')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--max_episodes', type=int, default=640, help='maximum number of episodes')
parser.add_argument('--max_steps', type=int, default=1000, help='maximum steps per episode')
parser.add_argument('--save_interval', type=int, default=20, help='interval to save model')
parser.add_argument('--target_update_interval', type=int, default=10, help='interval to update target network')
parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon for exploration')
parser.add_argument('--epsilon_end', type=float, default=0.05, help='Minimum epsilon value')
parser.add_argument('--epsilon_decay', type=float, default=0.985, help='Epsilon decay rate')


args = parser.parse_args()

replay_buffer_size = args.batch_size
hidden_dim = 64
hypernet_dim = 128
max_steps = args.max_steps
max_episodes = args.max_episodes
batch_size = args.batch_size
save_interval = args.save_interval
target_update_interval = args.target_update_interval
model_path = 'model/qmix'
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = battle_v4.env(
    map_size=45,
    minimap_mode=False,
    extra_features=False,
)
state_dim = 80
action_dim = env.action_space("blue_0").n
action_shape = 1
env.reset()
n_agents = len(env.agents)//2

replay_buffer = ReplayBufferGRU(replay_buffer_size)
learner = QMix_Trainer(
    replay_buffer, 
    n_agents, 
    state_dim, 
    action_shape, 
    action_dim, 
    hidden_dim, 
    hypernet_dim, 
    target_update_interval,
    epsilon_start=args.epsilon_start,
    epsilon_end=args.epsilon_end,
    epsilon_decay=args.epsilon_decay
)

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
    learner.mixer.train()
    learner.target_mixer.train()
    loss = None
    for episode in range(max_episodes):
        env.reset()
        episode_reward = 0
        
        # Clear memory after each episode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                   
        # Initialize hidden states for all blue agents
        hidden_states = torch.zeros(1, 1, n_agents, hidden_dim).to(device)
        ini_hidden_states = hidden_states.clone()
        
        # Lists to store episode data
        episode_states = []
        episode_actions = []
        episode_last_actions = []
        episode_rewards = []
        episode_next_states = []
        
        # Initialize last actions as zeros
        last_actions = np.zeros((n_agents, action_shape))
        # Init dead agents list
        dead_agents = []
        
        for step in range(max_steps):
            if all(env.truncations.values()):
                print(step)
                print("Environment truncated!!!")
                break
            # Get all blue agents states and rewards
            states, rewards, terminations, truncations, infos = get_all_states(env, dead_agents)
            if len(states) == 0:  # No blue agents alive
                break
            states = np.stack(states) # [n_agents, state_dim]

            # Get actions from RNNAgent
            actions, hidden_states = learner.get_action(states, last_actions, hidden_states)

            # Execute actions and collect next states/rewards
            next_states = []
            rewards = []
            # Save dead agents after making actions
            next_states, rewards, terminations, truncations, infos, dead_agents = make_action(actions, env, dead_agents)
            if len(next_states) == 0:  # No blue agents alive
                break

            next_states = np.stack(next_states) # [n_agents, state_dim]
            rewards = np.stack(rewards) # [n_agents]

            # Store transition
            episode_states.append(states)
            episode_actions.append(actions)
            episode_last_actions.append(last_actions)
            episode_rewards.append(rewards)
            episode_next_states.append(next_states)
            
            episode_reward += rewards.sum()
            last_actions = actions
            
        # print(np.stack(episode_states).shape) #(1000, 81, 845)
        # print(np.stack(episode_actions).shape) #(1000, 81, 1)
        # print(np.stack(episode_last_actions).shape) #(1000, 81, 1)
        # print(np.stack(episode_rewards).shape) #(1000, 81)
        # print(np.stack(episode_next_states).shape) #(1000, 81, 845)

        # Push entire episode to replay buffer
        if len(episode_states) > 0:
            learner.push_replay_buffer(
                ini_hidden_states,
                hidden_states,
                episode_states,
                episode_actions,
                episode_last_actions,
                episode_rewards,
                episode_next_states
            )
        
        # Clear unnecessary tensors
        del hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Training step
        if len(replay_buffer) >= batch_size:
            loss = learner.update(batch_size)

        # Save model periodically
        if episode % save_interval == 0:
            learner.save_model(f"{model_path}_episode_{episode}")
            
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Loss = {loss if loss else 'N/A'}")
    
    # Save final model
    learner.save_model(model_path)
    env.close()
    
    return learner

if __name__ == "__main__":
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