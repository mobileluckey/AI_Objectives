# Objective 6 – Project 2: RL Game Bot (Q-learning)

## Why this meets Objective 6
This project designs and integrates reinforcement learning using Q-learning. The agent learns a policy by maximizing reward through repeated interaction with an environment, demonstrating a core intelligent-software approach used in robotics and automation.

This project meets Objective 6 by demonstrating reinforcement learning using Q-learning to train an agent through trial and error. I built a small grid-based game where the bot starts in the top-left and must reach a goal in the bottom-right while avoiding a trap tile. The agent learns a policy by updating a Q-table, which stores expected rewards for each action in each state, and it improves over thousands of episodes using exploration (epsilon-greedy) and reward-based updates. After training, the bot consistently chooses a smarter path that reaches the goal and avoids the trap, showing how reinforcement learning can create intelligent behavior without hard-coded rules.

## Run
```bash
pip install -r requirements.txt
cd src
python qlearn_grid_bot.py
