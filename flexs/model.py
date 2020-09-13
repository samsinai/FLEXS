import abc

import flexs
from typing import Any, List

class Model(flexs.Landscape, abc.ABC):
    """Base structure for all models."""

    @abc.abstractmethod
    def train(self, sequences: Union[List[str], np.ndarray], labels: List[Any]):
        """Train model.

        This function is called whenever you would want your model to update itself based on the set of sequences it has measurements for."""
        pass
