import tensorflow as tf

from . import keras_model


class CNN(keras_model.KerasModel):
    def __init__(
        self,
        seq_len,
        num_filters,
        hidden_size,
        alphabet,
        loss="MSE",
        name=None,
        batch_size=256,
        epochs=20,
    ):

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv1D(
                    num_filters,
                    len(alphabet) - 1,
                    padding="valid",
                    strides=1,
                    input_shape=(seq_len, len(alphabet)),
                ),
                tf.keras.layers.Conv1D(
                    num_filters, 20, padding="same", activation="relu", strides=1
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
            model, alphabet=alphabet, name=name, batch_size=batch_size, epochs=epochs,
        )
