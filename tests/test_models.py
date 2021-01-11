import numpy as np
import pytest
import sklearn

import flexs
from flexs import baselines

rng = np.random.default_rng()


class FakeModel(flexs.Model):
    def _fitness_function(self, sequences):
        return rng.random(size=len(sequences))

    def train(self, *args, **kwargs):
        pass


class FakeLandscape(flexs.Landscape):
    def _fitness_function(self, sequences):
        return rng.random(size=len(sequences))


class FakeConstantModel(flexs.Model):
    def __init__(self, constant):
        super().__init__(name="ConstantModel")
        self.constant = constant

    def _fitness_function(self, sequences):
        return np.ones(len(sequences)) * self.constant

    def train(self, *args, **kwargs):
        pass


def test_adaptive_ensemble():
    models = [FakeConstantModel(1), FakeConstantModel(2)]
    ens = baselines.models.AdaptiveEnsemble(models)

    assert np.sum(ens.weights) == 1

    assert ens.get_fitness(["ATC"]) == 1.5

    models = [FakeModel(name="FakeModel") for _ in range(2)]
    ens = baselines.models.AdaptiveEnsemble(models)

    ens.train(["ATC"] * 15, list(range(15)))

    print(ens.weights)
    assert np.any(ens.weights != np.ones(len(models)) / len(models))
    # Possible floating-point error from summation
    assert np.isclose(np.sum(ens.weights), 1)


def test_keras_models():
    cnn = baselines.models.CNN(
        seq_len=3,
        num_filters=1,
        hidden_size=1,
        kernel_size=2,
        alphabet=flexs.utils.sequence_utils.DNAA,
    )
    cnn.get_fitness(["ATC"])

    gem = baselines.models.GlobalEpistasisModel(
        seq_len=3,
        hidden_size=1,
        alphabet=flexs.utils.sequence_utils.DNAA,
    )
    gem.get_fitness(["ATC"])

    mlp = baselines.models.MLP(
        seq_len=3,
        hidden_size=1,
        alphabet=flexs.utils.sequence_utils.DNAA,
    )
    mlp.get_fitness(["ATC"])


def test_noisy_abstract_model():
    nam = baselines.models.NoisyAbstractModel(
        landscape=FakeLandscape(name="FakeLandscape")
    )
    assert len(nam.cache) == 0
    fitness = nam.get_fitness(["ATC"])
    assert len(nam.cache) == 1
    assert nam.get_fitness(["ATC"]) == fitness

    nam = baselines.models.NoisyAbstractModel(
        landscape=FakeConstantModel(2), signal_strength=1
    )
    assert nam.get_fitness(["ATC"]) == [2]

    nam = baselines.models.NoisyAbstractModel(
        landscape=FakeConstantModel(2), signal_strength=0
    )
    nam.get_fitness(["ATC"])
    # Flaky, but extremely unlikely to fail
    assert nam.get_fitness(["ATG"]) != [2]


def test_sklearn_models():
    sklearn_models = [
        baselines.models.LinearRegression,
        baselines.models.LogisticRegression,
        baselines.models.RandomForest,
    ]
    for model in sklearn_models:
        m = model(
            alphabet=flexs.utils.sequence_utils.DNAA,
        )
        with pytest.raises(sklearn.exceptions.NotFittedError):
            m.get_fitness(["ATC"])
        m.train(["ATC", "ATG"], [1, 2])
        m.get_fitness(["ATC"])
