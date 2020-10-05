"""Define scikit-learn model wrappers as well a few convenient pre-wrapped models."""
import abc

import numpy as np
import sklearn.ensemble
import sklearn.linear_model

import flexs
from flexs.utils import sequence_utils as s_utils


class SklearnModel(flexs.Model, abc.ABC):
    """Base sklearn model wrapper."""

    def __init__(self, model, alphabet, name):
        """
        Args:
            model: sklearn model to wrap.
            alphabet: Alphabet string.
            name: Human-readable short model descriptipon (for logging).

        """
        super().__init__(name)

        self.model = model
        self.alphabet = alphabet

    def train(self, sequences, labels):
        """Flatten one-hot sequences and train model using `model.fit`."""
        one_hots = np.array(
            [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences]
        )
        flattened = one_hots.reshape(
            one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2]
        )
        self.model.fit(flattened, labels)


class SklearnRegressor(SklearnModel, abc.ABC):
    """Class for sklearn regressors (uses `model.predict`)."""

    def _fitness_function(self, sequences):
        one_hots = np.array(
            [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences]
        )
        flattened = one_hots.reshape(
            one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2]
        )

        return self.model.predict(flattened)


class SklearnClassifier(SklearnModel, abc.ABC):
    """Class for sklearn classifiers (uses `model.predict_proba(...)[:, 1]`)."""

    def _fitness_function(self, sequences):
        one_hots = np.array(
            [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences]
        )
        flattened = one_hots.reshape(
            one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2]
        )

        return self.model.predict_proba(flattened)[:, 1]


class LinearRegression(SklearnRegressor):
    """Sklearn linear regression."""

    def __init__(self, alphabet, **kwargs):
        """Create linear regression model."""
        model = sklearn.linear_model.LinearRegression(**kwargs)
        super().__init__(model, alphabet, "linear_regression")


class LogisticRegression(SklearnRegressor):
    """Sklearn logistic regression."""

    def __init__(self, alphabet, **kwargs):
        """Create logistic regression model."""
        model = sklearn.linear_model.LogisticRegression(**kwargs)
        super().__init__(model, alphabet, "logistic_regression")


class RandomForest(SklearnRegressor):
    """Sklearn random forest regressor."""

    def __init__(self, alphabet, **kwargs):
        """Create random forest regressor."""
        model = sklearn.ensemble.RandomForestRegressor(**kwargs)
        super().__init__(model, alphabet, "random_forest")
