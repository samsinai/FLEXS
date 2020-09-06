import abc

import flexs


class Model(flexs.Landscape, abc.ABC):
    """Base structure for all models."""

    @abc.abstractmethod
    def train(self, sequences, labels):
        """Train model.

        This function is called whenever you would want your model to update itself based on the set of sequences it has measurements for."""
        pass
