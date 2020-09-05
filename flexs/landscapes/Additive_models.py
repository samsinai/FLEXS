import json
import random
import sys

import numpy as np
from meta.model import Ground_truth_oracle
from utils.multi_dimensional_model import Multi_dimensional_model

AAV2_WT = """MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLD\
KGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQ\
AKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDAD\
SVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVI\
TTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLI\
NNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQG\
CLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPF\
HSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPG\
PCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVL\
IFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGV\
LPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTT\
FSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVY\
SEPRPIGTRYLTRNL"""


class Additive_landscape_constructor:
    def __init__(self):
        self.loaded_landscapes = {}

    def load_landscapes(
        self,
        config_file="../data/AAV_Additive_landscapes/landscapes_450_540.json",
        landscapes_to_test=None,
    ):

        with open(config_file, "r") as infile:
            all_loaded_landscapes = json.load(infile)

        if landscapes_to_test:
            for loaded_landscape in all_loaded_landscapes:
                if (
                    all_loaded_landscapes[loaded_landscape]["phenotype"]
                    in landscapes_to_test
                ):
                    self.loaded_landscapes[loaded_landscape] = all_loaded_landscapes[
                        loaded_landscape
                    ]

        else:
            self.loaded_landscapes = all_loaded_landscapes
            print(f"loaded {len(self.loaded_landscapes)} landscapes")

    def construct_landscape_object(self, landscape_id, landscape):

        additive_landscape = Additive_landscape_AAV(**landscape)
        seq_start = landscape["start"]
        seq_end = landscape["end"]

        return {
            "landscape_id": landscape_id,
            "starting_seqs": {
                f"AAV2:{seq_start}-{seq_end}": AAV2_WT[seq_start:seq_end]
            },
            "landscape_oracle": additive_landscape,
        }

    def generate_from_loaded_landscapes(self):

        for landscape_id in self.loaded_landscapes:

            yield self.construct_landscape_object(
                landscape_id, self.loaded_landscapes[landscape_id]
            )


class Additive_landscape_AAV(Ground_truth_oracle):
    def __init__(
        self,
        config_file="../data/AAV_Additive_landscapes/AAV2_single_subs.json",
        phenotype="heart",
        noise=0,
        minimum_fitness_multiplier=1,
        start=0,
        end=735,
    ):

        self.sequences = {}
        self.phenotype = f"log2_{phenotype}_v_wt"

        self.noise = noise
        self.mfm = minimum_fitness_multiplier
        self.start = start
        self.end = end
        with open(config_file, "r") as infile:
            self.loaded_data = json.load(infile)

        self.data = {}
        for (
            key
        ) in (
            self.loaded_data
        ):  # convert keys to integers and restrict them to the scope
            new_key = int(key)
            if new_key >= self.start and new_key < self.end:
                self.data[new_key] = self.loaded_data[key]

        self.top_seq, self.max_possible = self.compute_max_possible()

    def compute_max_possible(self):
        best_seq = ""
        max_fitness = 0
        for pos in self.data:
            current_max = -10
            current_best = "M"
            for aa in self.data[pos]:
                current_fit = self.data[pos][aa][self.phenotype]
                if (
                    current_fit > current_max
                    and self.data[pos][aa]["log2_packaging_v_wt"] > -6
                ):
                    current_best = aa
                    current_max = current_fit

            best_seq += current_best
            max_fitness += current_max
        return best_seq, max_fitness

    def _get_raw_fitness(self, seq):
        total_fitness = 0
        for i, s in enumerate(seq):
            if (
                s not in self.data[i + self.start]
                or self.data[i + self.start][s]["log2_packaging_v_wt"] == -6
            ):
                return 0
            else:
                total_fitness += self.data[i + self.start][s][self.phenotype]

        return total_fitness + self.mfm * self.max_possible

    def _fitness_function(self, seq):
        normed_fitness = self._get_raw_fitness(seq) / (
            self.max_possible * (self.mfm + 1)
        )
        return max(0, normed_fitness)

    def get_fitness(self, sequence):
        if self.noise == 0:
            if sequence in self.sequences:
                return self.sequences[sequence]
            else:
                self.sequences[sequence] = self._fitness_function(sequence)
                return self.sequences[sequence]
        else:
            self.sequences[sequence] = max(
                0,
                min(
                    1,
                    self._fitness_function(sequence)
                    + np.random.normal(scale=self.noise),
                ),
            )
        return self.sequences[sequence]
