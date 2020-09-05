import numpy as np

import flsd


class NullModel(flsd.Model):
    def __init__(self, cache=True, seed=None):
        self.average_fitness = 0.05
        self.rng = np.random.default_rng()

    def train(self, data):
        self.average_fitness = np.mean(data.y)

    def get_fitness(self, sequences):
        return self.rng.exponential(scale=self.average_fitness, size=len(sequences))