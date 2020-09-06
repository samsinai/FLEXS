import flexs
import numpy as np


class Ensemble(flexs.Model):
    def __init__(self, models, combine_with=lambda x: np.mean(x, axis=1)):
        name = f"Ens({'|'.join(model.name for model in models)})"
        super().__init__(name)

        self.models = models
        self.combine_with = combine_with

    def train(self, sequences, labels):
        for model in self.models:
            model.train(sequences, labels)

    def _fitness_function(self, sequences):
        scores = np.stack(
            [model.get_fitness(sequences) for model in self.models], axis=1
        )

        return self.combine_with(scores)
