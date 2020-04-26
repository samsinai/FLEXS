import numpy as np
import random
import yaml
import pyrosetta as prs
prs.init()
from meta.model import Ground_truth_oracle
from utils.multi_dimensional_model import Multi_dimensional_model


class Protein_landscape_constructor:
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

        for target in landscape_params["targets"]:
            landscapes.append(Protein_landscape_binding(target))

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

        return {
            "landscape_id": landscape_id,
            "starting_seqs": landscape_params["starts"],
            "landscape_oracle": composite_landscape,
        }

    def generate_from_loaded_landscapes(self):

        for landscape in self.loaded_landscapes:

            yield self.construct_landscape_object(landscape)



class Protein_landscape_folding(Ground_truth_oracle):
    def __init__(self, threshold=False, noise=0, norm_value=1, reverse=False, score_method=prs.get_fa_scorefxn()):
        self.sequences = {}
        self.noise = noise
        self.threshold = threshold
        self.norm_value = norm_value
        self.reverse = reverse
        self.score_method = score_method 

    def _fitness_function(self, sequence):
        pose = prs.pose_from_sequence(sequence)
        score = self.score_method(pose)
        return score / self.norm_value

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


class Protein_landscape_binding(Ground_truth_oracle):
    def __init__(self, target, threshold=False, noise=0, norm_value=1):
        self.target = target
        self.sequences = {}
        self.threshold = threshold
        self.noise = noise
        self.norm_value = self.compute_maximum_binding_possible(self.target)




