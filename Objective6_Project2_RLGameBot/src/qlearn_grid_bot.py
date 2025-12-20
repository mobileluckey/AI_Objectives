"""
Objective 6 – Project 2: RL Game Bot (Q-Learning)

What this program is:
- A tiny reinforcement learning agent that learns how to move in a grid world.
- Start: top-left
- Goal: bottom-right (big positive reward)
- Trap: one “bad” tile (big negative reward)

Why this matters for Objective 6:
- Reinforcement Learning (RL) is a core intelligent-software technique used in robotics,
  automation, and game AI.
- Q-learning learns a policy by trial-and-error: actions that lead to rewards become
  more likely, actions that lead to penalties become less likely.

Proof you'll see:
- A BEFORE training run (random-ish / greedy on empty Q-table)
- An AFTER training run (learned policy reaching the goal reliably)
- A saved text log in /models so you can commit proof to GitHub.
"""

from __future__ import annotations
import os
import numpy as np

# Actions the agent can choose from
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Map each action to a row/col movement (dr, dc)
MOVE = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def ensure_models_dir() -> str:
    """
    Create (if needed) and return the absolute path to the project's /models folder.
    This keeps proof outputs in the repo for GitHub and your website.
    """
    here = os.path.dirname(__file__)  # .../src
    models_dir = os.path.abspath(os.path.join(here, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def step(state: tuple[int, int],
         action: str,
         grid_size: int,
         goal: tuple[int, int],
         trap: tuple[int, int]) -> tuple[tuple[int, int], float, bool]:
    """
    Environment transition function.

    Inputs:
      state: (row, col)
      action: one of ACTIONS
      grid_size: width/height of square grid
      goal: terminal goal position
      trap: terminal trap position

    Outputs:
      next_state, reward, done
    """
    r, c = state
    dr, dc = MOVE[action]

    # Clamp movement so the agent can't leave the grid
    nr = max(0, min(grid_size - 1, r + dr))
    nc = max(0, min(grid_size - 1, c + dc))
    next_state = (nr, nc)

    # Terminal states: trap or goal
    if next_state == trap:
        return next_state, -10.0, True
    if next_state == goal:
        return next_state, +10.0, True

    # Small step penalty to encourage shorter paths
    return next_state, -0.1, False


def rollout_policy(Q: np.ndarray,
                   grid_size: int,
                   start: tuple[int, int],
                   goal: tuple[int, int],
                   trap: tuple[int, int],
                   max_steps: int = 30) -> list[str]:
    """
    Run the greedy policy implied by Q (always pick argmax action).

    Returns a list of log strings that you can print or save.
    """
    logs = []
    s = start

    for i in range(max_steps):
        # Choose the best-known action at this state
        a = int(np.argmax(Q[s[0], s[1]]))

        ns, reward, done = step(s, ACTIONS[a], grid_size, goal, trap)

        logs.append(f"step {i:02d}: state={s} action={ACTIONS[a]:>5} -> {ns} reward={reward:+.1f}")

        s = ns
        if done:
            if s == goal:
                logs.append("DONE: reached GOAL ✅")
            elif s == trap:
                logs.append("DONE: hit TRAP ❌")
            break

    return logs


def main() -> None:
    rng = np.random.default_rng(42)

    # Grid configuration
    grid_size = 5
    start = (0, 0)
    goal = (4, 4)
    trap = (3, 2)

    # Q-table:
    # Q[row, col, action_index] = expected future reward for taking that action
    Q = np.zeros((grid_size, grid_size, len(ACTIONS)), dtype=float)

    # Learning hyperparameters
    alpha = 0.2   # learning rate: how fast we update Q-values
    gamma = 0.95  # discount factor: value of future reward vs immediate reward
    eps = 1.0     # epsilon for epsilon-greedy exploration

    # Training loop settings
    episodes = 2500
    max_steps_per_episode = 80

    models_dir = ensure_models_dir()
    log_path = os.path.join(models_dir, "rl_qlearning_run.txt")

    # ----------------------------
    # BEFORE training: demo policy
    # ----------------------------
    # Q is all zeros, so argmax is always action 0 ("UP") which is NOT smart.
    before_logs = ["=== BEFORE TRAINING (empty Q-table) ==="]
    before_logs += rollout_policy(Q, grid_size, start, goal, trap, max_steps=30)
    before_logs.append("")

    # ----------------------------
    # TRAINING: Q-learning
    # ----------------------------
    progress_lines = ["=== TRAINING LOG (every 250 episodes) ==="]
    for ep in range(episodes):
        s = start

        for _ in range(max_steps_per_episode):
            # Epsilon-greedy:
            # - with probability eps, explore a random action
            # - otherwise, exploit the best-known action
            if rng.random() < eps:
                a_idx = int(rng.integers(0, len(ACTIONS)))
            else:
                a_idx = int(np.argmax(Q[s[0], s[1]]))

            # Step the environment
            ns, reward, done = step(s, ACTIONS[a_idx], grid_size, goal, trap)

            # Q-learning update:
            # Q(s,a) <- (1-a)*Q(s,a) + a*(reward + gamma*max_a' Q(s',a'))
            best_next = float(np.max(Q[ns[0], ns[1]]))
            Q[s[0], s[1], a_idx] = (1 - alpha) * Q[s[0], s[1], a_idx] + alpha * (reward + gamma * best_next)

            s = ns
            if done:
                break

        # Slowly reduce exploration so the agent becomes more confident over time
        eps = max(0.05, eps * 0.997)

        # Light progress logging (good for screenshots)
        if (ep + 1) % 250 == 0:
            progress_lines.append(f"episode={ep+1:04d} eps={eps:.3f}")

    progress_lines.append("")

    # ----------------------------
    # AFTER training: learned policy
    # ----------------------------
    after_logs = ["=== AFTER TRAINING (learned greedy policy) ==="]
    after_logs += rollout_policy(Q, grid_size, start, goal, trap, max_steps=30)

    # Combine logs, print them, and save them
    all_logs = before_logs + progress_lines + after_logs

    print("\n".join(all_logs))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_logs))

    print(f"\nSaved proof log to: {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
