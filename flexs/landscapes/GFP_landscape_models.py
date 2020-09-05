import os

import numpy as np
import pandas as pd
import requests
import torch
from meta.model import Ground_truth_oracle
from tape import ProteinBertForValuePrediction, TAPETokenizer
from tape.datasets import FluorescenceDataset
from tape.training import run_train
from utils.sequence_utils import *


def clean_GFP_df(df):
    # remove all invalid amino acids
    aa_set = set(AAS)
    df["seq"] = df["seq"].str.strip("*")
    df = df[df.apply(lambda x: set(x["seq"]).issubset(AAS), axis=1)]
    return df


class GFP_landscape_constructor:
    def __init__(self):
        self.loaded_landscapes = []
        self.starting_seqs = {}

    def load_landscapes(
        self, data_path="../data/GFP_landscapes", landscapes_to_test=None
    ):
        self.landscape_file = (
            f"{data_path}/sarkisyan_full_aa_seq_and_brightness_no_truncations.tsv"
        )
        self.all_seqs = clean_GFP_df(pd.read_csv(self.landscape_file, sep=","))
        self.all_seqs = self.all_seqs.sort_values("brightness").reset_index(drop=True)
        # starting sequences will be deciles of dataset
        if landscapes_to_test != "all":
            if landscapes_to_test:
                if type(landscapes_to_test[0]) == int:
                    self.starting_seqs = {
                        i: self.all_seqs.loc[
                            int(self.all_seqs.shape[0] * i / 10), "seq"
                        ]
                        for i in landscapes_to_test
                    }

        print(f"{len(self.starting_seqs)} GFP landscapes loaded.")

    def construct_landscape_object(self):
        landscape_id = "GFP"
        landscape = GFP_landscape()
        landscape.construct(self.all_seqs)

        return {
            "landscape_id": landscape_id,
            "starting_seqs": self.starting_seqs,
            "landscape_oracle": landscape,
        }

    def generate_from_loaded_landscapes(self):
        yield self.construct_landscape_object()


class GFP_landscape(Ground_truth_oracle):
    def __init__(self, noise=0, norm_value=1):
        """
        Green fluorescent protein (GFP) lanscape. The oracle used in this lanscape is
        the transformer model from TAPE (https://github.com/songlab-cal/tape).

        To create the transformer model used here, run the command:

        tape-train transformer fluorescence --from_pretrained bert-base --batch_size 128 --gradient_accumulation_steps 10 --data_dir .
        """
        self.sequences = {}
        self.noise = noise
        self.norm_value = norm_value
        self.gfp_model_path = "https://fluorescence-model.s3.amazonaws.com/fluorescence_transformer_20-05-25-03-49-06_184764/"
        self.save_path = "fluorescence-model/"
        os.makedirs(self.save_path, exist_ok=True)
        for file_name in [
            "args.json",
            "checkpoint.bin",
            "config.json",
            " log",
            "pytorch_model.bin",
        ]:
            if not os.path.isfile(self.save_path + file_name):
                print("Downloading", file_name)
                response = requests.get(self.gfp_model_path + file_name)
                with open(self.save_path + file_name, "wb") as f:
                    f.write(response.content)

    def construct(self, all_seqs):
        self.GFP_df = all_seqs
        self.GFP_info = dict(zip(self.GFP_df["seq"], self.GFP_df["brightness"]))
        self.tokenizer = TAPETokenizer(vocab="iupac")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ProteinBertForValuePrediction.from_pretrained(self.save_path).to(
            self.device
        )

    def _fitness_function(self, sequence):
        encoded_seq = torch.tensor([self.tokenizer.encode(sequence)]).to(self.device)
        (fitness_val,) = self.model(encoded_seq)
        return float(fitness_val)

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
