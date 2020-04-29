import numpy as np
import pandas as pd 
import random
import yaml
from glob import glob
import pyrosetta as prs
prs.init()
from meta.model import Ground_truth_oracle
from utils.multi_dimensional_model import Multi_dimensional_model


class Protein_landscape_constructor:
    def __init__(self):
        self.loaded_landscapes = []
        self.starting_seqs = {}

    def load_landscapes(self, data_path="../data/Protein_landscapes", landscapes_to_test=[]):
        df_info = pd.read_csv(f"{data_path}/Protein_landscape_id.csv", sep=",")
        start_seqs, target_seqs = {}, {}
        for _, row in df_info.iterrows():
            if row['id'] not in landscapes_to_test:
                continue 
            if row['id'].startswith('PF'):
                # protein folding landscape 
                native_pose = prs.pose_from_pdb(data_path + '/folding_landscapes/' + row['file_name'])
                seq = native_pose.sequence()
                self.starting_seqs[row['id']] = 'A' * len(seq)
                self.loaded_landscapes.append({'landscape_id': row['id'], 'targets': [seq]})
            else:
                # protein binding landscape 
                native_pose = prs.pose_from_pdb(data_path + '/binding_landscapes/' + row['file_name'])
                seq = native_pose.sequence()
                self.starting_seqs[row['id']] = 'A' * len(seq)
                self.loaded_landscapes.append({'landscape_id': row['id'], 'targets': [seq]})

        print(f"{len(self.loaded_landscapes)} Protein landscapes loaded.")

    def construct_landscape_object(self, landscape_dict):
        landscape_id = landscape_dict['landscape_id']
        landscapes = []
        for target in landscape_dict["targets"]:
            if landscape_id.startswith('PF'):
                landscapes.append(Protein_landscape_folding(target))
            else:
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
                "starting_seqs": self.starting_seqs,
                "landscape_oracle": landscapes[0],
            }

        return {
            "landscape_id": landscape_id,
            "starting_seqs": self.starting_seqs,
            "landscape_oracle": composite_landscape,
        }

    def generate_from_loaded_landscapes(self):

        for landscape in self.loaded_landscapes:

            yield self.construct_landscape_object(landscape)



class Protein_landscape_folding(Ground_truth_oracle):
    def __init__(self, target, noise=0, norm_value=1):
        self.sequences = {}
        self.target = target 
        self.target_pose = prs.pose_from_sequence(target) 
        self.noise = noise
        self.norm_value = norm_value

    def _fitness_function(self, sequence):
        pose = prs.pose_from_sequence(sequence)
        score = -prs.rosetta.protocols.stepwise.modeler.align.superimpose_with_stepwise_aligner(pose, self.target_pose)
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
    def __init__(self, target, noise=0, norm_value=1):
        self.target = target
        self.sequences = {}
        self.noise = noise
        self.norm_value = self.compute_maximum_binding_possible(self.target)




