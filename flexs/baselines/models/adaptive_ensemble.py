"""Defines the AdaptiveEnsemble model."""
from typing import List

import numpy as np
import scipy.stats
import sklearn.model_selection

import flexs
from flexs.types import SEQUENCES_TYPE


def r2_weights(model_preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Args:
        model_preds: A numpy array of shape (num_models, num_samples) containing
            model predictions.
        labels: A numpy array of true labels.

    Returns:
        A numpy array of shape (num_models,) containing $r^2$ scores for models.

    """
    r2s = np.array(
        [scipy.stats.pearsonr(preds, labels)[0] ** 2 for preds in model_preds]
    )
    return r2s / r2s.sum()


class AdaptiveEnsemble(flexs.Model):
    """
    Ensemble class that weights individual model predictions adaptively,
    according to some reweighting function.
    """

    def __init__(
        self,
        models: List[flexs.Model],
        combine_with="sum",
        adapt_weights_with="r2_weights",
        adaptive_val_size: float = 0.2,
    ):
        """
        Args:
            models: Models to ensemble
            combine_with: A function taking in weight vector and model outputs and
                returning an aggregate output per sample. `np.sum(weights * outputs)`
                by default.
            adapt_weights_with: A function taking in a numpy array of shape
                (num_models, num_samples) containing model predictions, and a numpy
                array of true labels (num_samples,) that returns an array of
                shape (num_models,) containing model_weights. `r2_weights` by default.
            adaptive_val_size: Portion of model training data to go into validation
                split used for computing adaptive weight values.
        """
        name = f"AdaptiveEns({'|'.join(model.name for model in models)})"
        super().__init__(name)

        self.models = models
        self.weights = np.ones(len(models)) / len(models)

        if combine_with == "sum":
            combine_with = lambda w, x: np.sum(w * x, axis=1)
        self.combine_with = combine_with

        if adapt_weights_with == "r2_weights":
            adapt_weights_with = r2_weights
        self.adapt_weights_with = adapt_weights_with

        self.adaptive_val_size = adaptive_val_size

    def train(self, sequences: SEQUENCES_TYPE, labels: np.ndarray):
        """
        Train each model in the ensemble and then adaptively reweight them
        according to `adapt_weights_with`.

        Args:
            sequences: Training sequences.
            lables: Training sequence labels.

        """
        # If very few sequences, don't bother with reweighting
        if len(sequences) < 10:
            for model in self.models:
                model.train(sequences, labels)
            return

        (train_X, test_X, train_y, test_y,) = sklearn.model_selection.train_test_split(
            np.array(sequences), np.array(labels), test_size=self.adaptive_val_size
        )

        for model in self.models:
            model.train(train_X, train_y)

        preds = np.stack([model.get_fitness(test_X) for model in self.models], axis=0)
        self.weights = self.adapt_weights_with(preds, test_y)

    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        scores = np.stack(
            [model.get_fitness(sequences) for model in self.models], axis=1
        )

        return self.combine_with(self.weights, scores)
