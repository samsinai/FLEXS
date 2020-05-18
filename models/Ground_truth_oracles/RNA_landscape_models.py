import sys

sys.path.append("usr/local/ViennaRNA/lib/python3.6/site-packages/")
# TODO: ANONYMIZE
sys.path.append("/n/home01/ssinaei/sw/lib/python3.4/site-packages/")
sys.path.append("/n/home01/ssinaei/sw/lib/python3.6/site-packages/")

import RNA
import numpy as np
import random
import yaml
from meta.model import Ground_truth_oracle
from utils.multi_dimensional_model import Multi_dimensional_model


class RNA_landscape_constructor:
    def __init__(self):
        self.loaded_landscapes = []

    def load_landscapes(self, config_file, landscapes_to_test=None):
        with open(config_file) as cfgfile:
            self.loaded_landscapes = yaml.load(cfgfile)

        if landscapes_to_test != "all":
            if landscapes_to_test:
                self.loaded_landscapes = [
                    self.loaded_landscapes[i] for i in landscapes_to_test
                ]
            else:
                self.loaded_landscapes = []

        for landscape_dict in self.loaded_landscapes:
            print(f"{list(landscape_dict.keys())[0]} loaded")

    def construct_landscape_object(self, landscape_dict):

        landscape_id = list(landscape_dict.keys())[0]
        landscape_params = landscape_dict[landscape_id]
        landscapes = []
        if (
            landscape_params["self_fold_max"] != False
        ): 
            """
            This is currently not used for any landscape, the idea would be to pass
            an upper bound on self-folding, beyond which the sequence is considered
            unviable.
            """
            l = RNA_landscape_folding(
                threshold=landscape_params["self_fold_max"], reverse=True
            )
            landscapes.append(l)

        for target in landscape_params["targets"]:
            landscapes.append(RNA_landscape_binding(target))

        if len(landscapes) > 1:
            from functools import reduce

            combined_func = lambda z: reduce(
                lambda x, y: x * y, z
            )  # currently multiplicative landscapes
            composite_landscape = Multi_dimensional_model(
                landscapes, combined_func=combined_func
            )

        else:
            return {
                "landscape_id": landscape_id,
                "starting_seqs": landscape_params["starts"],
                "landscape_oracle": landscapes[0],
            }

        if landscape_params["conserved_pattern"]:
            composite_landscape = Conserved_RNA_landscape_cont(
                composite_landscape,
                landscape_params["conserved_pattern"],
                landscape_params["conserved_start"],
            )

        return {
            "landscape_id": landscape_id,
            "starting_seqs": landscape_params["starts"],
            "landscape_oracle": composite_landscape,
        }

    def generate_from_loaded_landscapes(self):

        for landscape in self.loaded_landscapes:

            yield self.construct_landscape_object(landscape)


class RNA_landscape_folding(Ground_truth_oracle):
    def __init__(self, threshold=False, noise=0, norm_value=1, reverse=False):
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


class RNA_landscape_binding(Ground_truth_oracle):
    """An RNA binding landscape"""

    def __init__(self, target, threshold=False, noise=0, norm_value=1):
        self.target = target
        self.sequences = {}
        self.threshold = threshold
        self.noise = noise
        self.norm_value = self.compute_maximum_binding_possible(self.target)

    def _fitness_function(self, sequence):
        duplex = RNA.duplexfold(self.target, sequence)
        fitness = -duplex.energy
        if self.threshold != False:
            if fitness > self.threshold:
                return 1
            else:
                return 0

        return fitness / (self.norm_value * len(sequence) / len(self.target))

    def compute_maximum_binding_possible(self, target):
        map1 = {"A": "U", "C": "G", "G": "C", "U": "A"}
        match = ""
        for x in target:
            match += map1[x]
        dupenergy = RNA.duplexfold(match[::-1], target)
        return -dupenergy.energy

    def get_fitness(self, sequence):
        if type(sequence) == list:
            sequence = "".join(sequence)
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


class Conserved_RNA_landscape_cont(Ground_truth_oracle):
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


class Conserved_RNA_landscape_random(Ground_truth_oracle):
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
            return self.sequences[sequence]
