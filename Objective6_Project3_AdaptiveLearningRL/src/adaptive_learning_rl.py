"""
Objective 6 – Project 3: Adaptive Learning RL (Q-Learning Tutor)

What this program does:
- Simulates an adaptive learning system (like a mini TutiTech tutor).
- The "agent" chooses question difficulty: EASY / MEDIUM / HARD.
- The "student" has a hidden skill level that changes over time.
- The agent learns a policy using Reinforcement Learning (Q-learning) to maximize learning.

Key idea:
- If difficulty is too easy, the student learns very little.
- If difficulty is too hard, the student gets it wrong and progress slows.
- The RL agent learns the sweet spot for each skill level.

Why this fits Objective 6:
- This is reinforcement learning applied to an intelligent tutoring system.
- It uses state, action, reward, exploration (epsilon-greedy), and Q-value updates.
- It demonstrates adaptation based on feedback, which is a core intelligent-software behavior.

Outputs:
- Terminal output showing training progress + learned policy
- Saved proof log in /models/adaptive_learning_rl_log.txt
"""

from __future__ import annotations
import os
import numpy as np

# Actions represent "difficulty" choices the tutor can pick.
ACTIONS = ["EASY", "MEDIUM", "HARD"]

# Map difficulty to a numeric level for simulation rules.
DIFF_LEVEL = {"EASY": 0, "MEDIUM": 1, "HARD": 2}


def ensure_models_dir() -> str:
    """
    Create (if needed) and return absolute path to /models folder.
    We save proof logs here so it's easy to commit to GitHub.
    """
    here = os.path.dirname(__file__)  # .../src
    models_dir = os.path.abspath(os.path.join(here, "..", "models"))
    acknowledging = os.makedirs(models_dir, exist_ok=True)
    return models_dir


def student_step(skill: int,
                 action_idx: int,
                 rng: np.random.Generator) -> tuple[int, float, bool]:
    """
    Simulated student + environment response.

    State:
      skill in {0,1,2,3,4}
        0 = beginner
        4 = advanced

    Action:
      action_idx in {0,1,2} meaning EASY/MEDIUM/HARD

    Returns:
      next_skill, reward, done

    Reward shaping:
    - Big reward if the student answers correctly AND difficulty is appropriate.
    - Small reward for easy correct answers (low learning gain).
    - Penalty for incorrect answers (frustration / wasted time).
    - Additional small reward when skill increases (actual learning happened).

    Episode ends when:
    - skill reaches max (4) OR a fixed number of steps in main loop.
    """
    difficulty = action_idx
    diff_name = ACTIONS[action_idx]

    # Probability of correct answer depends on how difficulty compares to skill.
    # If difficulty <= skill: likely correct
    # If difficulty > skill: less likely correct
    #
    # This is a simplified model of learning: when tasks are slightly challenging
    # but not impossible, growth is best.
    gap = difficulty - min(2, skill)  # compare to skill but cap at 2 to match difficulty scale

    if gap <= 0:
        p_correct = 0.85  # student likely succeeds
    elif gap == 1:
        p_correct = 0.55  # somewhat hard
    else:
        p_correct = 0.25  # too hard

    correct = rng.random() < p_correct

    # Base reward: correctness signal
    if correct:
        # Learning gain is bigger when the question is not trivial
        # (medium/hard is more valuable than always-easy).
        base_reward = 0.4 + 0.3 * difficulty  # EASY=0.4, MEDIUM=0.7, HARD=1.0
    else:
        base_reward = -0.6  # penalty for incorrect answer

    # Skill update rule:
    # - If correct on MEDIUM or HARD, higher chance to increase skill.
    # - If incorrect on HARD when low skill, risk of stagnation.
    next_skill = skill

    if correct:
        # growth chance depends on difficulty (harder questions can promote growth)
        grow_chance = 0.25 + 0.15 * difficulty  # EASY=0.25, MEDIUM=0.40, HARD=0.55
        if rng.random() < grow_chance and skill < 4:
            next_skill += 1
    else:
        # We won't decrease skill here (keeps it simple), but we could.
        next_skill = skill

    # Extra reward for actual skill growth (this is what we truly want)
    reward = base_reward + (1.0 if next_skill > skill else 0.0)

    done = (next_skill >= 4)
    return next_skill, float(reward), done


def train_q_learning(episodes: int = 4000,
                     max_steps: int = 25,
                     seed: int = 42) -> tuple[np.ndarray, dict]:
    """
    Train a Q-learning policy.

    Q[state, action] = expected future reward

    Returns:
      Q-table, training summary metrics
    """
    rng = np.random.default_rng(seed)

    n_states = 5              # skill 0..4
    n_actions = len(ACTIONS)  # EASY/MEDIUM/HARD

    Q = np.zeros((n_states, n_actions), dtype=float)

    # Hyperparameters (tuned for stable learning + quick runtime)
    alpha = 0.25   # learning rate
    gamma = 0.95   # discount factor
    eps = 1.0      # exploration rate

    # Track progress for proof
    rewards_per_ep = []

    for ep in range(episodes):
        skill = 0  # start as beginner each episode
        total_reward = 0.0

        for _ in range(max_steps):
            # Epsilon-greedy action selection
            if rng.random() < eps:
                action = int(rng.integers(0, n_actions))
            else:
                action = int(np.argmax(Q[skill]))

            next_skill, reward, done = student_step(skill, action, rng)

            # Q-learning update rule:
            # Q(s,a) <- (1-a)*Q(s,a) + a*(reward + gamma*max_a' Q(s',a'))
            best_next = float(np.max(Q[next_skill]))
            Q[skill, action] = (1 - alpha) * Q[skill, action] + alpha * (reward + gamma * best_next)

            total_reward += reward
            skill = next_skill

            if done:
                break

        rewards_per_ep.append(total_reward)

        # Decay exploration so the agent gradually exploits what it learned
        eps = max(0.05, eps * 0.997)

    # Summary stats
    recent = rewards_per_ep[-200:] if len(rewards_per_ep) >= 200 else rewards_per_ep
    metrics = {
        "episodes": episodes,
        "max_steps": max_steps,
        "avg_reward_last_200": float(np.mean(recent)),
        "min_reward_last_200": float(np.min(recent)),
        "max_reward_last_200": float(np.max(recent)),
    }

    return Q, metrics


def demonstrate_policy(Q: np.ndarray, seed: int = 7) -> list[str]:
    """
    Demonstrate a single run using the learned greedy policy (no exploration).
    Produces logs suitable for screenshot and saving.
    """
    rng = np.random.default_rng(seed)

    logs = []
    logs.append("=== DEMO RUN (greedy learned policy) ===")

    skill = 0
    for step_i in range(20):
        action = int(np.argmax(Q[skill]))
        next_skill, reward, done = student_step(skill, action, rng)

        logs.append(
            f"step {step_i:02d} | skill={skill} | pick={ACTIONS[action]:>6} "
            f"| reward={reward:+.2f} | next_skill={next_skill}"
        )

        skill = next_skill
        if done:
            logs.append("DONE: student reached skill level 4 ✅")
            break

    return logs


def policy_table(Q: np.ndarray) -> list[str]:
    """
    Create a readable policy summary:
    For each skill level, show the best action and Q-values.
    """
    lines = []
    lines.append("=== LEARNED POLICY TABLE ===")
    for skill in range(Q.shape[0]):
        best_action = int(np.argmax(Q[skill]))
        qvals = ", ".join([f"{ACTIONS[a]}={Q[skill,a]:+.2f}" for a in range(Q.shape[1])])
        lines.append(f"skill {skill}: best={ACTIONS[best_action]} | {qvals}")
    return lines


def main() -> None:
    models_dir = ensure_models_dir()
    log_path = os.path.join(models_dir, "adaptive_learning_rl_log.txt")

    # 1) Train
    Q, metrics = train_q_learning(episodes=4000, max_steps=25, seed=42)

    # 2) Build proof output
    out = []
    out.append("=" * 70)
    out.append("Objective 6 – Project 3: Adaptive Learning RL (Q-Learning Tutor)")
    out.append("=" * 70)
    out.append(f"Training metrics: {metrics}")
    out.append("")

    out += policy_table(Q)
    out.append("")
    out += demonstrate_policy(Q, seed=7)
    out.append("")
    out.append(f"Saved proof log to: {log_path}")
    out.append("Done.")

    # 3) Print + save proof
    print("\n".join(out))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
