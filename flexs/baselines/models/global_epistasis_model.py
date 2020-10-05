"""Define a global epistasis model."""
import tensorflow as tf

from . import keras_model


class GlobalEpistasisModel(keras_model.KerasModel):
    """
    Global epistasis model.

    Weighted sum of input features follow by several dense layers.
    A simple, but relatively uneffective nonlinear model.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        alphabet: int,
        loss="MSE",
        name: str = None,
        batch_size: int = 256,
        epochs: int = 20,
    ):
        """Create a global epistasis model."""
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    1, input_shape=(seq_len, len(alphabet)), activation="relu"
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
