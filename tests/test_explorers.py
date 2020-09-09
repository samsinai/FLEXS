from flexs.model import Model
from flexs.landscape import Landscape
import numpy as np

from flexs.baselines.explorers.adalead import Adalead
from flexs.baselines.explorers.DynaPPO_explorer import DynaPPO

rng = np.random.default_rng()


class FakeModel(Model):
    def _fitness_function(self, sequences):
        if isinstance(sequences, list) or isinstance(sequences, np.ndarray):
            return rng.random(size=len(sequences))
        return rng.random()

    def train(self, *args, **kwargs):
        pass


class FakeLandscape(Landscape):
    def _fitness_function(self, sequences):
        if isinstance(sequences, list) or isinstance(sequences, np.ndarray):
            return rng.random(size=len(sequences))
        return rng.random()


fakeModel = FakeModel(name="FakeModel")
fakeLandscape = FakeLandscape(name="FakeLandscape")


def test_adalead():
    explorer = Adalead(
        model=fakeModel,
        landscape=fakeLandscape,
        rounds=1,
        sequences_batch_size=2,
        model_queries_per_batch=1,
        starting_sequence="A",
        alphabet="ATCG",
    )

    sequences, _ = explorer.run()
    print(sequences)

    # See @TODOs in adalead.py


def test_dynappo():
    explorer = DynaPPO(
        model=fakeModel,
        landscape=fakeLandscape,
        rounds=1,
        sequences_batch_size=4,
        model_queries_per_batch=8,
        starting_sequence="A",
        alphabet="ATCG",
        batch_size=1,
        threshold=0.5,
        num_experiment_rounds=1,
        num_model_rounds=1,
    )

    sequences, _ = explorer.run()
    print(sequences)


test_adalead()
test_dynappo()
