# Objective 6 – Project 3: Adaptive Learning System Using RL

## Why this meets Objective 6
This project integrates reinforcement learning concepts into an intelligent tutoring workflow. The system learns which vocabulary words to prioritize by maximizing reward (correct answers) through repeated interaction, demonstrating RL-driven adaptation for personalized learning.

This project meets Objective 6 by using reinforcement learning to build an adaptive learning system that changes its behavior based on student performance. I implemented a Q-learning tutor that selects question difficulty (easy, medium, hard) while a simulated student’s skill level changes over time. The agent learns a policy through trial and error using rewards, where correct answers and skill growth increase reward while incorrect answers reduce it. Over many episodes, the Q-table updates to favor difficulty choices that produce the best learning gains for each skill level, demonstrating adaptive decision-making instead of hard-coded rules. The final output includes the learned policy table and a demo run showing the tutor guiding the student toward mastery, which reflects how reinforcement learning can support personalized intelligent software.

## Run
```bash
pip install -r requirements.txt
cd src
python adaptive_learning_rl.py
