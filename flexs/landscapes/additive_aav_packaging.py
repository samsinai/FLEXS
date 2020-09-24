import json
import os

import numpy as np

import flexs


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


class AdditiveAAVPackaging(flexs.Landscape):
    def __init__(
        self, phenotype="heart", minimum_fitness_multiplier=1, start=0, end=735
    ):
        super().__init__(f"AdditiveAAVPackaging_phenotype={phenotype}")

        self.sequences = {}
        self.phenotype = f"log2_{phenotype}_v_wt"

        self.mfm = minimum_fitness_multiplier
        self.start = start
        self.end = end
        self.wild_type = AAV2_WT[start:end]

        with open(
            os.path.join(
                os.path.dirname(__file__),
                "data/additive_aav_packaging/AAV2_single_subs.json",
            )
        ) as f:
            self.data = {
                int(pos): val
                for pos, val in json.load(f).items()
                if self.start <= int(pos) < self.end
            }

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
            if s in self.data[self.start + i]:
                total_fitness += self.data[self.start + i][s][self.phenotype]

        return total_fitness + self.mfm * self.max_possible

    def _fitness_function(self, sequences):
        fitnesses = []
        for seq in sequences:
            normed_fitness = self._get_raw_fitness(seq) / (
                self.max_possible * (self.mfm + 1)
            )
            fitnesses.append(max(0, normed_fitness))

        return np.array(fitnesses)


def registry():
    """
    Returns a dictionary of problems of the form:
    `{
        "problem name": {
            "params": ...
        },
        ...
    }`

    where `flexs.landscapes.AdditiveAAVPackaging(**problem["params"])` instantiates the
    additive AAV packaging landscape for the given set of parameters.

    Returns:
        dict: Problems in the registry.

    """

    problems = {
        "heart": {"params": {"phenotype": "heart", "start": 450, "end": 540,}},
        "lung": {"params": {"phenotype": "lung", "start": 450, "end": 540}},
        "kidney": {"params": {"phenotype": "kidney", "start": 450, "end": 540}},
        "liver": {"params": {"phenotype": "liver", "start": 450, "end": 540}},
        "blood": {"params": {"phenotype": "blood", "start": 450, "end": 540}},
        "spleen": {"params": {"phenotype": "spleen", "start": 450, "end": 540}},
    }

    return problems
