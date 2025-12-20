import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduce TF noise in screenshots

import tensorflow as tf


def build_shallow_model(input_dim: int = 20) -> tf.keras.Model:
    """Shallow NN (1 hidden layer) baseline."""
    return tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,), name="features"),
        tf.keras.layers.Dense(16, activation="relu", name="hidden_1"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="ShallowNN")


def build_deep_model(input_dim: int = 20) -> tf.keras.Model:
    """Deeper NN (higher capacity, higher overfitting risk)."""
    return tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,), name="features"),
        tf.keras.layers.Dense(64, activation="relu", name="hidden_1"),
        tf.keras.layers.Dense(32, activation="relu", name="hidden_2"),
        tf.keras.layers.Dense(16, activation="relu", name="hidden_3"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="DeepNN")


def build_regularized_deep_model(input_dim: int = 20) -> tf.keras.Model:
    """Deep NN with Dropout regularization to improve generalization."""
    return tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,), name="features"),
        tf.keras.layers.Dense(64, activation="relu", name="hidden_1"),
        tf.keras.layers.Dropout(0.25, name="dropout_1"),
        tf.keras.layers.Dense(32, activation="relu", name="hidden_2"),
        tf.keras.layers.Dropout(0.15, name="dropout_2"),
        tf.keras.layers.Dense(16, activation="relu", name="hidden_3"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="DeepNN_Regularized")


def compile_for_demo(model: tf.keras.Model) -> None:
    """Compile to demonstrate optimizer/loss choices for binary classification."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


def try_save_diagram(model: tf.keras.Model, out_path: str) -> None:
    """
    Optional: save a model diagram (requires graphviz + pydot).
    If not installed, we fail gracefully so the project still works.
    """
    try:
        tf.keras.utils.plot_model(model, to_file=out_path, show_shapes=True, show_layer_names=True)
        print(f"Saved diagram: {out_path}")
    except Exception as e:
        print(f"(Optional) Diagram not saved for {model.name}.")
        print("Reason:", str(e))


def main():
    print("=" * 60)
    print("Objective 5 - Project 1: Neural Network Architecture Explainer")
    print("Folder: Objective5_Project1_NNFromScratch")
    print("=" * 60)

    input_dim = 20
    models = [
        build_shallow_model(input_dim),
        build_deep_model(input_dim),
        build_regularized_deep_model(input_dim)
    ]

    for m in models:
        compile_for_demo(m)

        print("\n" + "-" * 60)
        print(f"MODEL: {m.name}")
        print("-" * 60)
        m.summary()

        diagram_path = os.path.join("..", "assets", f"{m.name}.png")
        try_save_diagram(m, diagram_path)

    print("\nKey Takeaways:")
    print("1) More layers = more learning capacity, but also more overfitting risk.")
    print("2) ReLU adds non-linearity so the network can learn real patterns.")
    print("3) Sigmoid output gives a probability for binary classification.")
    print("4) Dropout is a best-practice regularization to improve generalization.")
    print("\nDone.")


if __name__ == "__main__":
    main()
