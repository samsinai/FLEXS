"""Define the base KerasModel class."""
from typing import Callable

import numpy as np
import tensorflow as tf

import flexs
from flexs.types import SEQUENCES_TYPE
from flexs.utils import sequence_utils as s_utils


class KerasModel(flexs.Model):
    """A wrapper around tensorflow/keras models."""

    def __init__(
        self,
        model,
        alphabet,
        name,
        batch_size=256,
        epochs=20,
        custom_train_function: Callable[[tf.Tensor, tf.Tensor], None] = None,
        custom_predict_function: Callable[[tf.Tensor], np.ndarray] = None,
    ):
        """
        Wrap a tensorflow/keras model.

        Args:
            model: A callable and fittable keras model.
            alphabet: Alphabet string.
            name: Human readable description of model (used for logging).
            batch_size: Batch size for `model.fit` and `model.predict`.
            epochs: Number of epochs to train for.
            custom_train_function: A function that receives a tensor of one-hot
                sequences and labels and trains `model`.
            custom_predict_function: A function that receives a tensor of one-hot
                sequences and predictions.

        """
        super().__init__(name)

        self.model = model
        self.alphabet = alphabet

        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size

    def train(
        self, sequences: SEQUENCES_TYPE, labels: np.ndarray, verbose: bool = False
    ):
        """Train keras model."""
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

        return np.nan_to_num(
            self.model.predict(one_hots, batch_size=self.batch_size).squeeze(axis=1)
        )
