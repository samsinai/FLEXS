import numpy as np
import scipy.stats

import flexs


def r2_weights(model_preds, labels):
    r2s = np.array(
        [scipy.stats.pearsonr(preds, labels)[0] ** 2 for preds in model_preds]
    )
    return r2s / r2s.sum()


class AdaptiveEnsemble(flexs.Model):
    def __init__(
        self,
        models,
        combine_with=lambda w, x: np.sum(w * x, axis=1),
        adapt_weights_with=r2_weights,
    ):
        name = f"AdaptiveEns({'|'.join(model.name for model in models)})"
        super().__init__(name)

        self.models = models
        self.weights = np.ones(len(models))

        self.combine_with = combine_with
        self.adapt_weights_with = adapt_weights_with

    def train(self, sequences, labels):
        for model in self.models:
            model.train(sequences, labels)

        preds = np.stack(
            [model.get_fitness(sequences) for model in self.models], axis=0
        )
        self.weights = self.adapt_weights_with(preds, labels)

    def _fitness_function(self, sequences):
        scores = np.stack(
            [model.get_fitness(sequences) for model in self.models], axis=1
        )

        return self.combine_with(self.weights, scores)
