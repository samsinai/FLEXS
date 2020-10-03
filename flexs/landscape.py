"""Defines the Landscape class."""
import abc

import numpy as np

from flexs.types import SEQUENCES_TYPE


class Landscape(abc.ABC):
    """
    Base class for all landscapes and for `flexs.Model`.

    Attributes:
        cost (int): Number of sequences whose fitness has been evaluated.
        name (str): A human-readable name for the landscape (often contains
            parameter values in the name) which is used when logging explorer runs.

    """

    def __init__(self, name: str):
        """Create Landscape, setting `name` and setting `cost` to zero."""
        self.cost = 0
        self.name = name

    @abc.abstractmethod
    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        pass

    def get_fitness(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        """
        Score a list/numpy array of sequences.

        This public method should not be overriden – new landscapes should
        override the private `_fitness_function` method instead. This method
        increments `self.cost` and then calls and returns `_fitness_function`.

        Args:
            sequences: A list/numpy array of sequence strings to be scored.

        Returns:
            Scores for each sequence.

        """
        self.cost += len(sequences)
        return self._fitness_function(sequences)
