from flexs.model import Model
from flexs.landscape import Landscape
import numpy as np

from flexs.baselines.explorers.adalead import Adalead

rng = np.random.default_rng()

class FakeModel(Model):
    def _fitness_function(self, sequences: np.ndarray):
        return rng.random(size=len(sequences))

    def train(self, *args, **kwargs):
        pass

class FakeLandscape(Landscape):
    def _fitness_function(self, sequences: np.ndarray):
        return rng.random(size=len(sequences))

fakeModel = FakeModel(name="FakeModel")
fakeLandscape = FakeLandscape(name="FakeLandscape")

def test_adalead():
    explorer = Adalead(
        model=fakeModel,
        landscape=fakeLandscape,
        rounds=1,
        initial_sequence_data=["A", "T"],
        experiment_budget=2,
        query_budget=1,
        alphabet="ATCG"
    )

    sequences, _ = explorer.run()

    # See @TODOs in adalead.py

test_adalead()