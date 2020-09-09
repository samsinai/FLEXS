import abc

import numpy as np
import sklearn.ensemble
import sklearn.linear_model

import flexs
from flexs.utils import sequence_utils as s_utils


class SklearnModel(flexs.Model, abc.ABC):
    def __init__(self, model, alphabet, loss, name, **kwargs):
        super().__init__(name)

        self.model = model
        self.alphabet = alphabet
        self.loss = loss

    def train(self, sequences, labels):
        one_hots = np.array(
            [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences]
        )
        flattened = one_hots.reshape(
            one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2]
        )
        self.model.fit(flattened, labels)

    def _fitness_function(self, sequences):
        one_hots = np.array(
            [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences]
        )
        flattened = one_hots.reshape(
            one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2]
        )

        if self.loss == "regression":
            return self.model.predict(flattened)
        elif self.loss == "classification":
            return self.model.predict_proba(flattened)[:, 1]


class LinearModel(SklearnModel):
    def __init__(self, alphabet, loss, **kwargs):
        if loss == "regression":
            model = sklearn.linear_model.LinearRegression(**kwargs)
        elif loss == "classification":
            model = sklearn.linear_model.LogisticRegression(**kwargs)
        else:
            raise ValueError('`loss` must be either "regression" or "classification"')

        super().__init__(model, alphabet, loss, f"linear_{loss}", **kwargs)


class RandomForest(SklearnModel):
    def __init__(self, alphabet, loss, **kwargs):
        if loss == "regression":
            model = sklearn.ensemble.RandomForestRegressor(**kwargs)
        elif loss == "classification":
            model = sklearn.ensemble.RandomForestClassifier(**kwargs)
        else:
            raise ValueError('`loss` must be either "regression" or "classification"')

        super().__init__(model, alphabet, loss, f"random_forest_{loss}", **kwargs)
