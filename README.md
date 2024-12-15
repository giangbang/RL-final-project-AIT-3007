# MAgent2 RL Final Project
## Overview
In this final project, you will develop and train a reinforcement learning (RL) agent using the MAgent2 platform. The task is to solve a specified MAgent2 environment `battle`, and your trained agent will be evaluated on all following three types of opponents:

1. Random Agents: Agents that take random actions in the environment.
2. A Pretrained Agent: A pretrained agent provided in the repository.
3. A Final Agent: A stronger pretrained agent, which will be released in the final week of the course before the deadline.

Your agent's performance should be evaluated based on reward and win rate against each of these models. You should control *blue* agents when evaluating.


<p align="center">
  <img src="assets/random.gif" width="300" alt="random agent" />
  <img src="assets/pretrained.gif" width="300" alt="pretrained agent" />
</p>

See `video` folder for a demo of how each type of opponent behaves.
Checkout a [Colab notebook](https://colab.research.google.com/drive/1qmx_NCmzPlc-atWqexn2WueqMKB_ZTxc?usp=sharing) for running this demo.

Update: The final ~~stronger~~ agent is released, this agent is trained on selfplay in about 15 minutes using DQN. See the below video

<p align="center">
<img src="assets/blueselfplay.gif" width="300" alt="selfplay blue vs random" />
  <img src="assets/redselfplay.gif" width="300" alt="selfplay combat" />
</p>
In the above demo, the left side shows the blue agent competing against random red agents, while the right side displays a battle between two self-play agents. Blue agents can comfortably defeat random agents, showing their cabability toward untrained agents, but they struggle with the red ones, which are intentionally trained more, so that they can dominate blue ones. As before, you should evaluate your agents against the red agents.

## Installation
clone this repo and install with
```
pip install -r requirements.txt
```

## Demos
See `main.py` for a starter code.

## Evaluation
Refer to `eval.py` for the evaluation code, you might want to modify it with your specific codebase.

## References

1. [MAgent2 GitHub Repository](https://github.com/Farama-Foundation/MAgent2)
2. [MAgent2 API Documentation](https://magent2.farama.org/introduction/basic_usage/)

For further details on environment setup and agent interactions, please refer to the MAgent2 documentation.
