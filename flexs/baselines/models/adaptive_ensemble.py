"""Defines the AdaptiveEnsemble model."""
import numpy as np
import scipy.stats

import flexs
from flexs.types import SEQUENCES_TYPE


class AdaptiveEnsemble(flexs.Model):
    def __init__(
        self,
        models,
        combine_with=lambda w, x: np.sum(w * x, axis=1),
        adapt_weights_with=None,
    ):
        name = f"AdaptiveEns({'|'.join(model.name for model in models)})"
        super().__init__(name)

        if adapt_weights_with is None:
            adapt_weights_with = self._r2_weights

        self.models = models
        self.weights = np.ones(len(models))

        self.combine_with = combine_with
        self.adapt_weights_with = adapt_weights_with

    def _r2_weights(model_preds, labels):
        r2s = np.array(
            [scipy.stats.pearsonr(preds, labels)[0] ** 2 for preds in model_preds]
        )
        return r2s / r2s.sum()

    def train(self, sequences: SEQUENCES_TYPE, labels: np.ndarray) -> np.ndarray:
        """
        Train each model in the ensemble and then adaptively reweight them
        according to their r^2 score.

        Args:
            sequences: Training sequences.
            lables: Training sequence labels.

        """
        for model in self.models:
            model.train(sequences, labels)

        preds = np.stack(
            [model.get_fitness(sequences) for model in self.models], axis=0
        )
        self.weights = self.adapt_weights_with(preds, labels)

    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        scores = np.stack(
            [model.get_fitness(sequences) for model in self.models], axis=1
        )

        return self.combine_with(self.weights, scores)
