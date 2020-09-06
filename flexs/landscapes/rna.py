import random

import flexs
import numpy as np
import RNA
import yaml


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


'''class Conserved_RNA_landscape_cont(Ground_truth_oracle):
    """Conserve a continous pattern positions along the sequence, mutating the pattern results in a dead sequence"""

    def __init__(self, RNAlandscape, pattern, start):
        self.sequences = {}
        self.pattern = pattern
        self.landscape = RNAlandscape
        self.start = start  # pattern could start here or later

    def _fitness_function(self, sequence):
        if self.pattern not in sequence[self.start :]:
            return 0
        else:
            return self.landscape.get_fitness(sequence)

    def get_fitness(self, sequence):
        if type(sequence) == list:
            sequence = "".join(sequence)

        if sequence in self.sequences:
            return self.sequences[sequence]
        else:
            self.sequences[sequence] = self._fitness_function(sequence)
            return self.sequences[sequence]


class Conserved_RNA_landscape_random(flexs.Landscape):
    """
    Conserve `n` random positions along the sequence;
    mutating them results in a dead sequence.
    """

    def __init__(self, RNAlandscape, wt, num_conserved):
        self.sequences = {}
        self.num_conserved = num_conserved
        self.landscape = RNAlandscape
        self.wt = wt
        random.seed(42)
        self.indices = random.sample(list(range(len(self.wt))), num_conserved)

    def _fitness_function(self, sequence):
        for i in self.indices:
            if self.wt[i] != sequence[i]:
                return 0
        else:
            return self.landscape.get_fitness(sequence)

    def get_fitness(self, sequence):
        if type(sequence) == list:
            sequence = "".join(sequence)

        if sequence in self.sequences:
            return self.sequences[sequence]
        else:
            self.sequences[sequence] = self._fitness_function(sequence)
            return self.sequences[sequence]'''
