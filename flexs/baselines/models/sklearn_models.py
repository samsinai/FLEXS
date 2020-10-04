import abc

import numpy as np
import sklearn.ensemble
import sklearn.linear_model

import flexs
from flexs.utils import sequence_utils as s_utils


class SklearnModel(flexs.Model, abc.ABC):
    def __init__(self, model, alphabet, name):
        super().__init__(name)

        self.model = model
        self.alphabet = alphabet

    def train(self, sequences, labels):
        one_hots = np.array(
            [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences]
        )
        flattened = one_hots.reshape(
            one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2]
        )
        self.model.fit(flattened, labels)


class SklearnRegressor(SklearnModel, abc.ABC):
    def _fitness_function(self, sequences):
        one_hots = np.array(
            [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences]
        )
        flattened = one_hots.reshape(
            one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2]
        )

        return self.model.predict(flattened)


class SklearnClassifier(SklearnModel, abc.ABC):
    def _fitness_function(self, sequences):
        one_hots = np.array(
            [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences]
        )
        flattened = one_hots.reshape(
            one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2]
        )

        return self.model.predict_proba(flattened)[:, 1]


class LinearRegression(SklearnRegressor):
    def __init__(self, alphabet, **kwargs):
        model = sklearn.linear_model.LinearRegression(**kwargs)
        super().__init__(model, alphabet, "linear_regression")


class LogisticRegression(SklearnRegressor):
    def __init__(self, alphabet, **kwargs):
        model = sklearn.linear_model.LogisticRegression(**kwargs)
        super().__init__(model, alphabet, "logistic_regression")


class RandomForest(SklearnRegressor):
    def __init__(self, alphabet, **kwargs):
        model = sklearn.ensemble.RandomForestRegressor(**kwargs)
        super().__init__(model, alphabet, "random_forest")
