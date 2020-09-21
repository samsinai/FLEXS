import numpy as np
import tensorflow as tf

import flexs
from flexs.utils import sequence_utils as s_utils


class KerasModel(flexs.Model):
    def __init__(
        self, model, alphabet, name, batch_size=256, epochs=20,
    ):
        super().__init__(name)

        self.model = model
        self.alphabet = alphabet

        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, sequences, labels, verbose=False):
        one_hots = tf.convert_to_tensor(
            np.array(
                [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences]
            ),
            dtype=tf.float32,
        )
        labels = tf.convert_to_tensor(labels)

        self.model.fit(
            one_hots,
            labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=verbose,
        )

    def _fitness_function(self, sequences):
        one_hots = tf.convert_to_tensor(
            np.array(
                [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences]
            ),
            dtype=tf.float32,
        )

        return self.model.predict(one_hots, batch_size=self.batch_size).squeeze(axis=1)
