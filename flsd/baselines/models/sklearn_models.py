import abc

import numpy as np
import sklearn.ensemble
import sklearn.linear_model

import flsd
import flsd.utils.sequence_utils as s_utils

class SklearnModel(flsd.Model, abc.ABC):

    def __init__(self, alphabet, loss, **kwargs):
        self.alphabet = alphabet
        self.loss = loss
        self.model = self.get_model(**kwargs)

    @abc.abstractmethod
    def get_model(self, **kwargs):
        pass

    def train(self, sequences, labels):
        one_hots = np.array([s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences])
        flattened = one_hots.reshape(one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2])
        self.model.fit(flattened, labels)

    def get_fitness(self, sequences):
        one_hots = np.array([s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences])
        flattened = one_hots.reshape(one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2])

        if self.loss == 'regression':
            return self.model.predict(flattened)
        elif self.loss == 'classification':
            return self.model.predict_proba(flattened)[:, 1]


class LinearModel(SklearnModel):

    def get_model(self, **kwargs):
        if self.loss == 'regression':
            return sklearn.linear_model.LinearRegression(**kwargs)
        elif self.loss == 'classification':
            return sklearn.linear_model.LogisticRegression(**kwargs)
        else:
            raise ValueError('`loss` must be either "regression" or "classification"')


class RandomForest(SklearnModel):

    def get_model(self, **kwargs):
        if self.loss == 'regression':
            return sklearn.ensemble.RandomForestRegressor(**kwargs)
        elif self.loss == 'classification':
            return sklearn.ensemble.RandomForestClassifier(**kwargs)
        else:
            raise ValueError('`loss` must be either "regression" or "classification"')