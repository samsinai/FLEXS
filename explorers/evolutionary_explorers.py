import random
from collections import Counter

import numpy as np

from explorers.base_explorer import Base_explorer
from utils.sequence_utils import generate_random_mutant


class Evolution(Base_explorer):
    def __init__(
        self,
        mu=1,
        rho=1,
        recomb_rate=0.0,
        beta=100,
        batch_size=100,
        alphabet="UCGA",
        virtual_screen=0,
        path="./simulations/",
        debug=False,
    ):
        super(Evolution, self).__init__(
            batch_size=batch_size,
            alphabet=alphabet,
            virtual_screen=virtual_screen,
            path=path,
            debug=debug,
        )
        self.mu = mu
        self.rho = rho
        self.recomb_rate = recomb_rate
        self.beta = beta

    def recombine(self, sequence1, sequence2):
        recomb_1 = []
        recomb_2 = []
        flipped = 1
        for s1, s2 in zip(sequence1, sequence2):
            if random.random() < self.recomb_rate:
                flipped = -flipped
            if flipped > 0:
                recomb_1.append(s1)
                recomb_2.append(s2)
            else:
                recomb_1.append(s2)
                recomb_2.append(s1)

        return "".join(recomb_1), "".join(recomb_2)

    def recombine_population(self, gen):
        random.shuffle(gen)
        ret = []
        for i in range(0, len(gen), 2):
            strA = []
            strB = []
            switch = False
            for ind in range(len(gen[i])):
                if random.random() < self.recomb_rate:
                    switch = not switch

                # putting together recombinants
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])

            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret

    def compute_fitnesses(self, sequences):
        pop_size = len(sequences)
        fitnesses = [0] * pop_size
        for i in range(pop_size):
            fitnesses[i] = np.exp(self.model.get_fitness(sequences[i]) * self.beta)
        fitnesses = np.array(fitnesses) / (sum(fitnesses))
        return fitnesses

    def propose_samples(self):
        """implement this function for your own explorer"""
        raise NotImplementedError(
            "propose_samples must be implemented by your explorer"
        )


class Moran(Evolution):
    def __init__(
        self,
        mu=0.01,
        rho=1,
        recomb_rate=0.0,
        beta=100,
        batch_size=100,
        alphabet="UCGA",
        virtual_screen=0,
    ):

        super(WF, self).__init__(
            mu, rho, recomb_rate, beta, batch_size, alphabet, virtual_screen
        )
        self.explorer_type = (
            f"Moran_mu{self.mu}_r{self.recomb_rate}_rho{self.rho}_beta{self.beta}"
        )

    def propose_samples(self):

        raise NotImplementedError(
            "`propose_samples` must be implemented by your explorer."
        )


class WF(Evolution):
    def __init__(
        self,
        mu=1,
        rho=1,
        recomb_rate=0.0,
        beta=100,
        batch_size=100,
        alphabet="UCGA",
        virtual_screen=0,
        path="./simulations/",
    ):
        super(WF, self).__init__(
            mu, rho, recomb_rate, beta, batch_size, alphabet, virtual_screen, path
        )
        self.explorer_type = (
            f"WF_mu{self.mu}_r{self.recomb_rate}_rho{self.rho}_beta{self.beta}"
        )

    def propose_samples(self):

        last_batch = self.get_last_batch()
        current_population = [seq for seq in self.batches[last_batch]]
        while len(current_population) < self.batch_size:
            current_population.append(current_population[0])

        fitnesses = self.compute_fitnesses(current_population)
        probabilities_from_fitness = np.cumsum(fitnesses)
        replicated_sequences = []
        for i in range(self.batch_size):
            sample = np.random.uniform()
            picked_sequence_index = np.searchsorted(probabilities_from_fitness, sample)
            seq = current_population[picked_sequence_index]
            new_sequence = generate_random_mutant(
                seq, self.mu / len(seq), alphabet=self.alphabet
            )
            replicated_sequences.append(new_sequence)

        if self.recomb_rate > 0:
            for i in range(self.rho):
                recombined_replicated_sequences_half = self.recombine_population(
                    replicated_sequences
                )[: self.batch_size]
        else:
            recombined_replicated_sequences_half = replicated_sequences

        return recombined_replicated_sequences_half


class ML_WF(Evolution):
    def __init__(
        self,
        mu=1,
        rho=1,
        recomb_rate=0.0,
        beta=100,
        batch_size=100,
        alphabet="UCGA",
        virtual_screen=20,
        path="./simulations/",
    ):
        super(ML_WF, self).__init__(
            mu, rho, recomb_rate, beta, batch_size, alphabet, virtual_screen, path
        )
        self.explorer_type = (
            f"MLWFG_mu{self.mu}_r{self.recomb_rate}_rho{self.rho}_beta{self.beta}"
        )

    def sub_sample(self, sequences):
        top_seqs_and_fits = []
        for seq in set(sequences):
            top_seqs_and_fits.append((self.model.get_fitness(seq), seq))

        top_seqs_and_fits = sorted(top_seqs_and_fits, reverse=True)
        return [t[1] for t in top_seqs_and_fits][: self.batch_size]

    def propose_samples(self):

        last_batch = self.get_last_batch()
        current_population = [seq for seq in self.batches[last_batch]]
        while len(current_population) < self.batch_size:
            current_population.append(current_population[0])

        fitnesses = self.compute_fitnesses(current_population)
        probabilities_from_fitness = np.cumsum(fitnesses)
        replicated_sequences = []
        while len(replicated_sequences) < self.batch_size * self.virtual_screen:
            sample = np.random.uniform()
            picked_sequence_index = np.searchsorted(probabilities_from_fitness, sample)
            seq = current_population[picked_sequence_index]
            new_sequence = generate_random_mutant(
                seq, self.mu / len(seq), alphabet=self.alphabet
            )
            replicated_sequences.append(new_sequence)

        if self.recomb_rate > 0:
            for i in range(self.rho):
                recombined_replicated_sequences_half = self.recombine_population(
                    replicated_sequences
                )
        else:
            recombined_replicated_sequences_half = replicated_sequences

        all_sequences = list(set(recombined_replicated_sequences_half))[
            : self.batch_size * self.virtual_screen
        ]

        selected_sequences = self.sub_sample(all_sequences)

        return selected_sequences
