"""
Objective 5 - Project 2: Decision Boundary Visualizer

Purpose:
- Demonstrate deep understanding of neural networks by VISUALIZING how different models
  draw decision boundaries.
- Compare a simple linear-ish model (Logistic Regression) vs. a Neural Network (MLP).
- Save plots to the /models folder so they can be uploaded to GitHub and used on a website.

Why this matters:
- A decision boundary is the "line" (or curve) where a model switches its prediction
  from Class 0 to Class 1.
- Logistic Regression typically learns a near-linear boundary.
- Neural Networks can learn non-linear boundaries by stacking hidden layers + activations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def plot_boundary(model, X, y, title, out_path):
    """
    Plot the decision boundary for a trained classifier and save it as a PNG.

    model:
        Any sklearn classifier that supports model.predict(X).
    X:
        Feature array shaped (n_samples, 2) so we can visualize it in 2D.
    y:
        Labels (0/1).
    title:
        Title shown at top of the plot.
    out_path:
        Where to save the PNG (example: ../models/boundary_mlp.png).
    """

    # 1) Define the viewing window slightly bigger than the data range.
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # 2) Create a grid of points that covers the window.
    #    We will ask the model to predict every point in this grid.
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # 3) Flatten the grid into a list of (x1, x2) points for prediction.
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 4) Predict class for each grid point, then reshape to match xx/yy.
    zz = model.predict(grid).reshape(xx.shape)

    # 5) Plot
    plt.figure(figsize=(7, 5))

    # Background color shows the region the model predicts as class 0 vs class 1
    plt.contourf(xx, yy, zz, alpha=0.25)

    # Overlay the actual data points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=25)

    # Labels + title
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")

    # Make sure layout doesn't cut off text
    plt.tight_layout()

    # 6) Save to disk for GitHub + website usage
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot: {out_path}")


def main():
    """
    Main workflow:
    1) Generate a non-linear dataset (two moons)
    2) Train Logistic Regression (mostly linear boundary)
    3) Train a Neural Network (non-linear boundary)
    4) Print accuracy for both
    5) Save both decision boundary plots as PNG files
    """

    print("=" * 60)
    print("Objective 5 - Project 2: Decision Boundary Visualizer")
    print("=" * 60)

    # Create output directory path (../models relative to src/)
    # This makes the script work even if launched from src folder.
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    # Output image files
    out_logreg = os.path.join(models_dir, "boundary_logreg.png")
    out_mlp = os.path.join(models_dir, "boundary_mlp.png")

    # 1) Make a dataset that needs a curved boundary to classify well
    #    "two moons" is famous because it's NOT linearly separable.
    X, y = make_moons(n_samples=600, noise=0.25, random_state=42)

    # 2) Train Logistic Regression
    #    Logistic Regression is a strong baseline but struggles with non-linear shapes.
    lr = LogisticRegression()
    lr.fit(X, y)

    # 3) Train a Neural Network (MLP = Multi-Layer Perceptron)
    #    Hidden layers + ReLU activation allow non-linear boundaries.
    nn = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=800,
        random_state=42
    )
    nn.fit(X, y)

    # 4) Print accuracy on training data (for demo purposes)
    #    This is not a perfect evaluation method, but good for visual proof.
    print(f"LogReg training accuracy: {lr.score(X, y):.3f}")
    print(f"MLP   training accuracy: {nn.score(X, y):.3f}")

    # 5) Save decision boundary plots
    plot_boundary(lr, X, y, "Decision Boundary: Logistic Regression (Linear-ish)", out_logreg)
    plot_boundary(nn, X, y, "Decision Boundary: Neural Network (Non-linear)", out_mlp)

    print("\nDone. Check the models folder for PNG images:")
    print(f" - {out_logreg}")
    print(f" - {out_mlp}")


if __name__ == "__main__":
    main()
