import numpy as np
import tensorflow as tf

import flsd
import flsd.utils.sequence_utils as s_utils

class KerasModel(flsd.Model):

    def __init__(
        self,
        model,
        alphabet,

        name=None,
        batch_size=256,
        epochs=20,
    ):
        self.model = model
        self.alphabet = alphabet

        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, sequences, labels, verbose=False):
        one_hots = np.array([s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences])

        self.model.fit(
            one_hots,
            labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=verbose
        )

    def get_fitness(self, sequences):
        one_hots = np.array([s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences])

        return self.model.predict(one_hots, batch_size=self.batch_size).squeeze(axis=1)
