import os

import numpy as np
import pandas as pd

import flexs


class TFBinding(flexs.Landscape):
    def __init__(self, landscape_file):
        super().__init__(name="TF_Binding")

        # Load TF pairwise TF binding measurements from file
        data = pd.read_csv(landscape_file, sep="\t")
        score = data["E-score"]
        norm_score = (score - score.min()) / (score.max() - score.min())

        # Populate dictionary with normalized scores
        self.sequences = dict(zip(data["8-mer"], norm_score))

    def _fitness_function(self, sequences):
        return np.array([self.sequences[seq] for seq in sequences])


def registry():
    """
    Returns a dictionary of problems of the form:
    `{
        "problem name": {
            "params": ...,
        },
        ...
    }`

    where `flexs.landscapes.TFBinding(**problem["params"])` instantiates the
    transcription factor binding landscape for the given set of parameters.

    Returns:
        dict: Problems in the registry.

    """

    tf_binding_data_dir = os.path.join(os.path.dirname(__file__), "data/tf_binding")

    problems = {}
    for fname in os.listdir(tf_binding_data_dir):
        problem_name = fname.replace("_8mers.txt", "")

        problems[problem_name] = {
            "params": {"landscape_file": os.path.join(tf_binding_data_dir, fname)}
        }

    return problems
