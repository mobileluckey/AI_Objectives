import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(y): return y * (1 - y)

def main():
    # XOR dataset (classic proof that we need non-linear hidden layers)
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)

    rng = np.random.default_rng(42)
    W1 = rng.normal(0, 1, (2, 4))
    b1 = np.zeros((1, 4))
    W2 = rng.normal(0, 1, (4, 1))
    b2 = np.zeros((1, 1))

    lr = 0.5
    for epoch in range(5000):
        # forward
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)

        # loss (MSE)
        loss = np.mean((y - a2) ** 2)

        # backward
        d_a2 = (a2 - y) * 2 / len(X)
        d_z2 = d_a2 * sigmoid_deriv(a2)
        dW2 = a1.T @ d_z2
        db2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = d_z2 @ W2.T
        d_z1 = d_a1 * sigmoid_deriv(a1)
        dW1 = X.T @ d_z1
        db1 = np.sum(d_z1, axis=0, keepdims=True)

        # update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if epoch % 500 == 0:
            print(f"epoch={epoch} loss={loss:.6f}")

    preds = (a2 >= 0.5).astype(int)
    print("Predictions:\n", preds.reshape(-1))
    print("Truth:\n", y.reshape(-1))

if __name__ == "__main__":
    main()
