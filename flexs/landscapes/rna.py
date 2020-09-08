import random

import numpy as np
import RNA
import yaml

import flexs


class RNAFolding(flexs.Landscape):
    def __init__(self, threshold=False, norm_value=1, reverse=False):
        super().__init__(name="RNAFolding")

        self.sequences = {}
        self.noise = noise
        self.threshold = threshold
        self.norm_value = norm_value
        self.reverse = reverse

    def _fitness_function(self, sequence):
        _, fe = RNA.fold(sequence)

        if self.threshold != False:
            if -fe > self.threshold:

                return int(not self.reverse)
            else:
                return int(self.reverse)

        return -fe / self.norm_value

    def get_fitness(self, sequence):
        if self.noise == 0:
            if sequence in self.sequences:
                return self.sequences[sequence]
            else:
                self.sequences[sequence] = self._fitness_function(sequence)
                return self.sequences[sequence]
        else:
            self.sequences[sequence] = self._fitness_function(
                sequence
            ) + np.random.normal(scale=self.noise)
        return self.sequences[sequence]


class RNABinding(flexs.Landscape):
    """An RNA binding landscape"""

    def __init__(self, target, threshold=False, noise=0):
        super().__init__(name="RNAFolding")

        self.target = target
        self.sequences = {}
        self.threshold = threshold
        self.noise = noise
        self.norm_value = self.compute_maximum_binding_possible(self.target)

    def compute_maximum_binding_possible(self, target):
        map1 = {"A": "U", "C": "G", "G": "C", "U": "A"}
        match = ""
        for x in target:
            match += map1[x]
        dupenergy = RNA.duplexfold(match[::-1], target)
        return -dupenergy.energy

    def _fitness_function(self, sequences):
        fitnesses = []

        for sequence in sequences:
            duplex = RNA.duplexfold(self.target, sequence)

            if self.threshold:
                fitness = int(-duplex.energy > self.threshold)
            else:
                fitness = -duplex.energy / (
                    self.norm_value * len(sequence) / len(self.target)
                )

            fitnesses.append(fitness)

        return np.array(fitnesses)
