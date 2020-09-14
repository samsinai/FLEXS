import numpy as np

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


fakeModel = FakeModel(name="FakeModel")
fakeLandscape = FakeLandscape(name="FakeLandscape")


def test_adalead():
    explorer = baselines.explorers.Adalead(
        model=fakeModel,
        landscape=fakeLandscape,
        rounds=3,
        sequences_batch_size=5,
        model_queries_per_batch=20,
        starting_sequence="ATC",
        alphabet="ATCG",
    )

    sequences, _ = explorer.run()
    print(sequences)

    # See @TODOs in adalead.py

def test_bo():
    explorer = baselines.explorers.BO_Explorer(
        model=fakeModel,
        landscape=fakeLandscape,
        rounds=1,
        ground_truth_measurements_per_round=2,
        model_queries_per_round=1,
        starting_sequence="A",
        alphabet="ATCG",
    )

    sequences, _ = explorer.run()
    print(sequences)

def test_dqn():
    explorer = baselines.explorers.DQN_Explorer(
        model=fakeModel,
        landscape=fakeLandscape,
        rounds=1,
        ground_truth_measurements_per_round=2,
        model_queries_per_round=1,
        starting_sequence="A",
        alphabet="ATCG",
    )

    sequences, _ = explorer.run()
    print(sequences)

def test_dynappo():
    explorer = baselines.explorers.DynaPPO(
        model=fakeModel,
        landscape=fakeLandscape,
        rounds=3,
        sequences_batch_size=5,
        model_queries_per_batch=20,
        starting_sequence="ATC",
        alphabet="ATCG",
        threshold=0.5,
        num_experiment_rounds=1,
        num_model_rounds=1,
    )

    sequences, _ = explorer.run()
    print(sequences)


def test_cmaes():
    explorer = baselines.explorers.CMAES(
        fakeModel,
        fakeLandscape,
        population_size=15,
        max_iter=200,
        initial_variance=0.3,
        rounds=3,
        starting_sequence="ATC",
        sequences_batch_size=5,
        model_queries_per_batch=20,
        alphabet="ATCG",
    )

    sequences, _ = explorer.run()
    print(sequences)
