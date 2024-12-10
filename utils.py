import numpy as np
from eval_qmix import get_pretrain_red_policy

blue_agents = ['blue_0', 'blue_1', 'blue_2', 'blue_3', 'blue_4', 'blue_5', 'blue_6', 'blue_7', 'blue_8', 'blue_9', 'blue_10', 'blue_11', 'blue_12', 'blue_13', 'blue_14', 'blue_15', 'blue_16', 'blue_17', 'blue_18', 'blue_19', 'blue_20', 'blue_21', 'blue_22', 'blue_23', 'blue_24', 'blue_25', 'blue_26', 'blue_27', 'blue_28', 'blue_29', 'blue_30', 'blue_31', 'blue_32', 'blue_33', 'blue_34', 'blue_35', 'blue_36', 'blue_37', 'blue_38', 'blue_39', 'blue_40', 'blue_41', 'blue_42', 'blue_43', 'blue_44', 'blue_45', 'blue_46', 'blue_47', 'blue_48', 'blue_49', 'blue_50', 'blue_51', 'blue_52', 'blue_53', 'blue_54', 'blue_55', 'blue_56', 'blue_57', 'blue_58', 'blue_59', 'blue_60', 'blue_61', 'blue_62', 'blue_63', 'blue_64', 'blue_65', 'blue_66', 'blue_67', 'blue_68', 'blue_69', 'blue_70', 'blue_71', 'blue_72', 'blue_73', 'blue_74', 'blue_75', 'blue_76', 'blue_77', 'blue_78', 'blue_79', 'blue_80']

def get_agent_states(env, agent):
    """
    Get state for a specific agent
    
    Args:
        env: MAgent environment
        agent: Agent ID
    
    Returns:
        observation, reward, termination, truncation, info for the agent
    """
    try:
        observation, reward, termination, truncation, info = env.observe(agent), env._cumulative_rewards[agent], env.terminations[agent], env.truncations[agent], env.infos[agent]
    except Exception as e:
        print(f"Error: {agent} - {e}")
        print(env.agents)
    
    return observation, reward, termination, truncation, info

def get_padding_states(env, agent):
    """
    Get padding states for a specific agent
    
    Args:
        env: MAgent environment
        agent: Agent ID
    
    Returns:
        Padding observation, reward, termination, truncation, info for the agent
    """
    observation, reward, termination, truncation, info = np.zeros((13, 13, 5)), 0, True, False, {}
    
    return observation, reward, termination, truncation, info

def get_all_states(env, dead_agents):
    """
    Get available states, rewards for all alive agent
    
    Args:
        env: MAgent environment
    
    Returns:
        List of available steps, rewards, terminations, truncations, infos for all alive blue agents
    """
    observations, rewards, terminations, truncations, infos = [], [], [], [], []
    for agent in blue_agents:
        if agent in dead_agents:
            observation, reward, termination, truncation, info = get_padding_states(env, agent)
        else:
            observation, reward, termination, truncation, info = get_agent_states(env, agent)

        observations.append(observation)
        rewards.append(reward)
        terminations.append(termination)
        truncations.append(truncation)
        infos.append(info)
    state = env.state()
    return observations, state, rewards, terminations, truncations, infos

def make_action(actions, env, dead_agents, red_agent=None):
    """
    Execute actions for all agents, including handling dead agents
    
    Args:
        actions: Array of actions: [n_agents, action_shape]
        env: MAgent environment
    Returns:
        Tuple of (observations, rewards, terminations, truncations, infos, dead_agents)
        for blue agents only
    """
    
    # action_shape=1 [81, 1] -> [81]
    actions = actions.reshape(-1)
    for _, agent in enumerate(env.agents):
        #Handle dead agents
        while agent == env.agents[0] and env.agent_selection != env.agents[0]:
            dead_agents.append(env.agent_selection)
            env.step(None)
            
        # Handle dead agents
        observation, _, termination, truncation, _ = env.last()
        if termination or truncation:
            env.step(action=None)
        else:
            # Blue agents move
            if agent.startswith("blue"):
                env.step(actions[blue_agents.index(agent)])
            # Random red agents move
            else:
                if red_agent is None:
                    env.step(env.action_space(agent).sample())
                else:
                    env.step(get_pretrain_red_policy(red_agent)(env, agent, observation))

    return dead_agents