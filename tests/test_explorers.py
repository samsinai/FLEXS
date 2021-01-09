import numpy as np

import flexs
from flexs import baselines


class FakeModel(flexs.Model):
    def _fitness_function(self, sequences):
        return np.random.random(size=len(sequences))

    def train(self, *args, **kwargs):
        pass


class FakeLandscape(flexs.Landscape):
    def _fitness_function(self, sequences):
        return np.random.random(size=len(sequences))


starting_sequence = "ATCATCAT"
fakeModel = FakeModel(name="FakeModel")
fakeLandscape = FakeLandscape(name="FakeLandscape")


def test_random():
    explorer = baselines.explorers.Random(
        model=fakeModel,
        rounds=3,
        sequences_batch_size=5,
        model_queries_per_batch=20,
        starting_sequence=starting_sequence,
        alphabet="ATCG",
    )
    explorer.run(fakeLandscape)


def test_adalead():
    explorer = baselines.explorers.Adalead(
        model=fakeModel,
        rounds=3,
        sequences_batch_size=5,
        model_queries_per_batch=20,
        eval_batch_size=1,
        starting_sequence=starting_sequence,
        alphabet="ATCG",
    )
    explorer.run(fakeLandscape)


def test_bo():
    explorer = baselines.explorers.BO(
        model=fakeModel,
        rounds=3,
        sequences_batch_size=5,
        model_queries_per_batch=20,
        starting_sequence=starting_sequence,
        alphabet="ATCG",
    )
    explorer.run(fakeLandscape)


def test_gpr_bo():
    explorer = baselines.explorers.GPR_BO(
        model=fakeModel,
        rounds=3,
        sequences_batch_size=5,
        model_queries_per_batch=20,
        starting_sequence=starting_sequence,
        alphabet="ATCG",
    )
    explorer.run(fakeLandscape)


def test_dqn():
    explorer = baselines.explorers.DQN(
        model=fakeModel,
        rounds=3,
        sequences_batch_size=5,
        model_queries_per_batch=20,
        starting_sequence=starting_sequence,
        alphabet="ATCG",
    )
    explorer.run(fakeLandscape)


def test_dynappo():
    explorer = baselines.explorers.DynaPPO(
        landscape=fakeLandscape,
        rounds=3,
        sequences_batch_size=5,
        model_queries_per_batch=20,
        starting_sequence=starting_sequence,
        alphabet="ATCG",
        num_experiment_rounds=1,
        num_model_rounds=1,
    )
    explorer.run(fakeLandscape)


def test_cmaes():
    explorer = baselines.explorers.CMAES(
        fakeModel,
        population_size=15,
        max_iter=200,
        initial_variance=0.3,
        rounds=3,
        starting_sequence=starting_sequence,
        sequences_batch_size=5,
        model_queries_per_batch=20,
        alphabet="ATCG",
    )
    explorer.run(fakeLandscape)


def test_cbas():
    vae = baselines.explorers.VAE(
        len(starting_sequence), "ATCG", epochs=2, verbose=False
    )
    explorer = baselines.explorers.CbAS(
        fakeModel,
        vae,
        rounds=3,
        starting_sequence=starting_sequence,
        sequences_batch_size=5,
        model_queries_per_batch=20,
        alphabet="ATCG",
    )
    explorer.run(fakeLandscape)
