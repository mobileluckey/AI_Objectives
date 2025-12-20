"""
Objective 6 – Project 1 (Reboot): Genetic Algorithm Parameter Tuner

What this program does:
- Uses a Genetic Algorithm (GA) to tune TWO hyperparameters for a classifier:
  1) Logistic Regression regularization strength (C)
  2) Probability decision threshold (thr)

Why this matters for Objective 6:
- Genetic Algorithms are a classic "evolutionary" optimization technique.
- This project demonstrates selection, crossover, mutation, and fitness evaluation.
- The GA searches a parameter space efficiently when brute-force tuning is expensive.

How to run:
  pip install -r requirements.txt
  cd src
  python ga_tuner.py

Expected output:
- Per-generation logs showing best F1 score improving over time
- Final best individual: C, threshold, and best F1 score
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Fitness function (GA objective)
# -----------------------------
def fitness(individual: np.ndarray,
            X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Compute "fitness" for one GA individual.
    The GA's goal is to maximize this value.

    individual structure:
      individual[0] = C        (regularization strength for LogisticRegression)
      individual[1] = thr      (probability threshold to convert prob -> class)

    Why F1 score:
    - Our dataset is slightly imbalanced (60/40), and F1 balances precision/recall
      better than raw accuracy.
    """
    C = float(individual[0])
    thr = float(individual[1])

    # Use a solver that is stable across many settings
    # and increase max_iter to avoid convergence warnings.
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=3000)

    # Train the model
    model.fit(X_train, y_train)

    # Predict probabilities then apply the GA threshold
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= thr).astype(int)

    return float(f1_score(y_test, preds))


# -----------------------------
# GA operators: mutation & crossover
# -----------------------------
def mutate(child: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Mutation step:
    - Slightly adjust C and threshold with random noise.
    - Clamp values so they stay in valid ranges.

    Mutation introduces variety so the GA doesn't get stuck.
    """
    child = child.copy()

    # Mutate C multiplicatively (keeps it positive and scales naturally)
    c_multiplier = float(np.clip(rng.normal(loc=1.0, scale=0.2), 0.6, 1.4))
    child[0] *= c_multiplier

    # Mutate threshold additively
    child[1] += float(rng.normal(loc=0.0, scale=0.05))

    # Clamp to valid ranges
    child[0] = float(np.clip(child[0], 0.01, 50.0))   # C must be > 0
    child[1] = float(np.clip(child[1], 0.05, 0.95))   # avoid extreme thresholds

    return child


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Crossover step:
    - Blend two parents to create a child.

    This is like mixing two "genetic" parameter sets together.
    """
    alpha = float(rng.random())  # blend ratio in [0,1]
    child_C = alpha * a[0] + (1 - alpha) * b[0]
    child_thr = alpha * a[1] + (1 - alpha) * b[1]
    return np.array([child_C, child_thr], dtype=float)


# -----------------------------
# Main GA loop
# -----------------------------
def main() -> None:
    rng = np.random.default_rng(42)

    # 1) Create a synthetic dataset
    #    make_classification is reliable and keeps the project self-contained.
    X, y = make_classification(
        n_samples=2500,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        weights=[0.6, 0.4],        # mild imbalance: makes F1 meaningful
        random_state=42
    )

    # 2) Train/test split (stratify preserves class ratio in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # 3) GA settings
    pop_size = 30
    generations = 25
    elite_keep = 6  # number of best individuals carried to next generation

    # 4) Initialize population
    #    Each individual is [C, threshold]
    #    - C in [0.01, 10]
    #    - threshold in [0.2, 0.8]
    population = np.column_stack([
        rng.uniform(0.01, 10.0, size=pop_size),
        rng.uniform(0.2, 0.8, size=pop_size)
    ]).astype(float)

    # Track the best solution found across all generations
    best_individual = None
    best_score = -1.0

    print("=" * 70)
    print("Objective 6 – Project 1: Genetic Algorithm Parameter Tuner (Reboot)")
    print("Tuning: LogisticRegression(C) + probability threshold (thr)")
    print("Fitness: F1 score on test split")
    print("=" * 70)

    # 5) Run evolution
    for gen in range(generations):
        # Evaluate all individuals (fitness for each)
        scores = np.array([
            fitness(ind, X_train, y_train, X_test, y_test)
            for ind in population
        ], dtype=float)

        # Select elites (top performers)
        elite_idx = scores.argsort()[::-1][:elite_keep]
        elite = population[elite_idx]

        # Update global best if improved
        if scores[elite_idx[0]] > best_score:
            best_score = float(scores[elite_idx[0]])
            best_individual = elite[0].copy()

        # Log progress (this is a GREAT screenshot for your website/GitHub proof)
        print(
            f"gen={gen:02d}  best_f1={best_score:.4f}  "
            f"best_C={best_individual[0]:.4f}  best_thr={best_individual[1]:.3f}"
        )

        # Build next generation
        next_population = [elite[i].copy() for i in range(len(elite))]

        # Fill the rest of the population with children
        while len(next_population) < pop_size:
            # Pick two random parents from elite pool
            i1, i2 = rng.integers(0, len(elite), size=2)
            p1, p2 = elite[i1], elite[i2]

            # Crossover + mutation
            child = crossover(p1, p2, rng)
            child = mutate(child, rng)

            next_population.append(child)

        population = np.array(next_population, dtype=float)

    # 6) Final report
    print("\nFinal Best Individual:")
    print(f"  C={best_individual[0]:.4f}")
    print(f"  threshold={best_individual[1]:.3f}")
    print(f"  best_F1={best_score:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
