import numpy as np
import pandas as pd

import flexs


class TF_binding_landscape_constructor:
    def __init__(self):
        self.loaded_landscapes = []
        self.starting_seqs = {}

    def load_landscapes(
        self, data_path="../data/TF_binding_landscapes", landscapes_to_test=None
    ):
        start_seqs = pd.read_csv(f"{data_path}/TF_starting_id.csv", sep=",")
        for i, row in start_seqs.iterrows():
            self.starting_seqs[row["id"]] = row["start"]

        self.loaded_landscapes = glob(f"{data_path}/landscapes/*")

        if landscapes_to_test != "all":
            if landscapes_to_test:
                if type(landscapes_to_test[0]) == int:
                    self.loaded_landscapes = [
                        self.loaded_landscapes[i] for i in landscapes_to_test
                    ]

                elif type(landscapes_to_test[0]) == str:
                    filtered_list = []
                    for landscape in landscapes_to_test:
                        for filename in self.loaded_landscapes:
                            if landscape in filename:
                                filtered_list.append(filename)

                    self.loaded_landscapes = filtered_list

            else:
                self.loaded_landscapes = []

        print(f"{len(self.loaded_landscapes)} TF landscapes loaded.")

    def construct_landscape_object(self, landscape_file):

        landscape_id = landscape_file.strip(
            "../data/TF_binding_landscapes/landscapes/"
        ).strip("_8mers.txt")
        landscape = TF_binding_landscape()
        landscape.construct(landscape_file)

        return {
            "landscape_id": landscape_id,
            "starting_seqs": self.starting_seqs,
            "landscape_oracle": landscape,
        }

    def generate_from_loaded_landscapes(self):

        for landscape_file in self.loaded_landscapes:

            yield self.construct_landscape_object(landscape_file)


class TFBinding(flexs.Landscape):
    def __init__(self, landscape_file):
        super().__init__(name="TF_Binding")

        # Load TF pairwise TF binding measurements from file
        data = pd.read_csv(landscape_file, sep="\t")
        score = data["Median"]
        norm_score = (score - score.min()) / (score.max() - score.min())

        # Populate dictionary with normalized scores
        self.sequences = dict(zip(data["8-mer"], norm_score))

    def _fitness_function(self, sequences):
        return np.array([self.sequences[seq] for seq in sequences])
