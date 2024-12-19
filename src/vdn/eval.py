def evaluate_model(env, num_episodes, model, run_episode_fn):
    """

    :param env: Environment
    :param num_episodes: How many episodes to test
    :param model: Trained model
    :param run_episode_fn: function to run an episode
    :return: average score over num_episodes
    """
    model.eval()
    score = 0
    for _ in range(num_episodes):
        score += run_episode_fn(env, model, epsilon=0)
    return score / num_episodes