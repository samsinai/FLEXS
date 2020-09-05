import numpy as np

import flsd

class Ensemble(flsd.Model):

    def __init__(self, models, combine_with=lambda x: np.mean(x, axis=1)):
        self.models = models
        self.combine_with = combine_with

    def train(self, sequences, labels):
        for model in self.models:
            model.train(sequences, labels)

    def get_fitness(self, sequences, combine_with=None):
        scores = np.stack([model.get_fitness(sequences) for model in self.models], axis=1)

        if combine_with is not None:
            return combine_with(scores)

        return self.combine_with(scores)