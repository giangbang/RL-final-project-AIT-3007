import sys
from magent2.environments import battle_v4
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
import torch
if __name__ == "__main__":
    env = battle_v4.env(map_size=45)
    env.reset()
    batch_size = 32

    
    # Initialize replay buffer with the correct field names and agent IDs
    rb = MultiAgentReplayBuffer(
        memory_size=100000,
        field_names=["obs", "action", "reward", "next_obs", "done"],
        agent_ids=env.possible_agents,  # Use possible_agents instead of agents
        device="cpu"
    )

    done = False
    count = 0
    observations = {}
    actions = {}
    rewards = {}
    next_observations = {}
    dones = {}
    # print(env.possible_agents)
    # sys.exit(1)

    for episode in range(2):
        while not done:
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                
                if termination or truncation:
                    action = None
                    done[agent] = True
                elif episode < 128:
                    action = env.action_space(agent).sample()
                else:
                    # implement the logic here
                    action = None
                
                # Store experience for current agent
                observations[agent] = observation
                actions[agent] = action
                rewards[agent] = reward
                # env.step(None)
                if not done:
                    env.step(action)
                    next_observation, reward, termination, truncation, info = env.last()
                    next_observations[agent] = next_observation
                else:
                    next_observation = None
                    
                # Save complete transition when we have data for all agents
                if len(observations) == len(env.possible_agents):
                    rb.save_to_memory_single_env(
                        observations.copy(),
                        actions.copy(), 
                        rewards.copy(),
                        next_observations.copy(),
                        dones.copy()
                    )
                    # Clear the dictionaries
                    observations.clear()
                    actions.clear()
                    rewards.clear()
                    next_observations.clear()
                    dones.clear()
                
                # Sample from buffer when we have enough experiences
                # create a if statement to check if len(memory) > batch_size, then
                # sample from a rb and update the network
                if len(rb):
                    samples = rb.sample(1)

                
                # count += 1
                # if count >= 20:
                #     break