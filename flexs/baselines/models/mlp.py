"""Define a baseline multilayer perceptron model."""
import tensorflow as tf

from . import keras_model


class MLP(keras_model.KerasModel):
    """A baseline MLP with three dense layers and relu activations."""

    def __init__(
        self,
        seq_len,
        hidden_size,
        alphabet,
        loss="MSE",
        name=None,
        batch_size=256,
        epochs=20,
    ):
        """Create an MLP."""
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    hidden_size, input_shape=(seq_len, len(alphabet)), activation="relu"
                ),
                tf.keras.layers.Dense(hidden_size, activation="relu"),
                tf.keras.layers.Dense(hidden_size, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        model.compile(loss=loss, optimizer="adam", metrics=["mse"])

        if name is None:
            name = f"MLP_hidden_size_{hidden_size}"

        super().__init__(
            model,
            alphabet=alphabet,
            name=name,
            batch_size=batch_size,
            epochs=epochs,
        )
