"""Define a baseline CNN Model."""
import tensorflow as tf

from . import keras_model


class CNN(keras_model.KerasModel):
    """A baseline CNN model with 3 conv layers and 2 dense layers."""

    def __init__(
        self,
        seq_len: int,
        num_filters: int,
        hidden_size: int,
        alphabet: str,
        loss="MSE",
        kernel_size: int = 5,
        name: str = None,
        batch_size: int = 256,
        epochs: int = 20,
    ):
        """Create the CNN."""
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv1D(
                    num_filters,
                    kernel_size,
                    padding="valid",
                    activation="relu",
                    strides=1,
                    input_shape=(seq_len, len(alphabet)),
                ),
                tf.keras.layers.Conv1D(
                    num_filters,
                    kernel_size,
                    padding="same",
                    activation="relu",
                    strides=1,
                ),
                tf.keras.layers.MaxPooling1D(1),
                tf.keras.layers.Conv1D(
                    num_filters,
                    len(alphabet) - 1,
                    padding="same",
                    activation="relu",
                    strides=1,
                ),
                tf.keras.layers.GlobalMaxPooling1D(),
                tf.keras.layers.Dense(hidden_size, activation="relu"),
                tf.keras.layers.Dense(hidden_size, activation="relu"),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(1),
            ]
        )

        model.compile(loss=loss, optimizer="adam", metrics=["mse"])

        if name is None:
            name = f"CNN_hidden_size_{hidden_size}_num_filters_{num_filters}"

        super().__init__(
            model,
            alphabet=alphabet,
            name=name,
            batch_size=batch_size,
            epochs=epochs,
        )
