from meta.model import Ground_truth_oracle
import numpy as np
import pandas as pd
import torch 
from tape import ProteinBertForValuePrediction, TAPETokenizer

from utils.sequence_utils import *

def clean_GFP_df(df):
    # remove all invalid amino acids 
    aa_set = set(AAS)
    df['seq'] = df['seq'].str.strip('*')
    df = df[df.apply(lambda x: set(x['seq']).issubset(AAS), axis=1)]
    return df

class GFP_landscape_constructor:
    def __init__(self):
        self.loaded_landscapes = []
        self.starting_seqs = {}

    def load_landscapes(
        self, data_path="../data/GFP_landscapes", landscapes_to_test=None
    ):
        self.landscape_file = f"{data_path}/sarkisyan_full_aa_seq_and_brightness_no_truncations.tsv"
        self.all_seqs = clean_GFP_df(pd.read_csv(self.landscape_file, sep=","))
        # hold out 10% for testing as in here: https://www.biorxiv.org/content/10.1101/337154v1.full.pdf
        self.held_out = self.all_seqs.sample(frac=0.1)
        self.all_seqs = self.all_seqs.drop(self.held_out.index) 
        self.all_seqs = self.all_seqs.sort_values('brightness')
        # starting sequences will be deciles of dataset 
        if landscapes_to_test != "all":
            if landscapes_to_test:
                if type(landscapes_to_test[0]) == int:
                    self.starting_seqs = {
                        i: self.all_seqs.loc[int(self.all_seqs.shape[0] * i / 10), 'seq'] for i in landscapes_to_test
                    }

        print(f"{len(self.starting_seqs)} GFP landscapes loaded.")

    def construct_landscape_object(self):
        landscape_id = 'GFP'
        landscape = GFP_landscape()
        landscape.construct(self.held_out, self.all_seqs)

        return {
            "landscape_id": landscape_id,
            "starting_seqs": self.starting_seqs,
            "landscape_oracle": landscape,
        }

    def generate_from_loaded_landscapes(self):
        yield self.construct_landscape_object()


class GFP_landscape(Ground_truth_oracle):
    def __init__(self, noise=0, norm_value=1):
        self.sequences = {}
        self.noise = noise
        self.norm_value = norm_value

    def construct(self, held_out, all_seqs):
        self.held_out = held_out
        self.GFP_df = all_seqs
        self.GFP_info = dict(zip(self.GFP_df['seq'], self.GFP_df['brightness']))
        self.model = ProteinBertForValuePrediction.from_pretrained('bert-base')
        self.tokenizer = TAPETokenizer(vocab='iupac')

    def _fitness_function(self, sequence):
        if sequence in self.GFP_info:
            return self.GFP_info[sequence]
        else:
            encoded_seq = torch.tensor([self.tokenizer.encode(sequence)])
            fitness_val, = self.model(encoded_seq)
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