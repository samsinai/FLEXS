"""Elitist explorers."""
import bisect
import random

import numpy as np
from flexs.baselines.explorers.base_explorer import Base_explorer
from flexs.utils.sequence_utils import generate_random_mutant
from flexs.utils.softmax import softmax


class XE_IS(Base_explorer):
    """
    Independent sites cross-entropy explorer.
    """

    def __init__(
        self,
        beta=1,
        recomb_rate=0,
        threshold=0.05,
        batch_size=100,
        alphabet="UCGA",
        virtual_screen=10,
        path="./simulations/",
        debug=False,
    ):
        """X-entropy independent-sites explorer."""
        super(XE_IS, self).__init__(
            batch_size=batch_size,
            alphabet=alphabet,
            virtual_screen=virtual_screen,
            path=path,
            debug=debug,
        )
        self.threshold = threshold
        self.beta = beta
        self.recomb_rate = recomb_rate
        self.explorer_type = (
            f"XE_IS_tr{self.threshold}_beta{self.beta}_r{self.recomb_rate}"
        )

    @staticmethod
    def recombine(sequence1, sequence2, rate):
        """Recombine."""
        recomb_1 = []
        recomb_2 = []
        flipped = 1
        for s1, s2 in zip(sequence1, sequence2):
            if random.random() < rate:
                flipped = -flipped
            if flipped > 0:
                recomb_1.append(s1)
                recomb_2.append(s2)
            else:
                recomb_1.append(s2)
                recomb_2.append(s1)

        return "".join(recomb_1), "".join(recomb_2)

    def get_matrix(self, top_seqs):
        """Get XE matrix."""
        XE_matrix = np.ones((len(self.alphabet), len(top_seqs[0])))

        for seq in top_seqs:
            for col, s in enumerate(seq):
                row = self.alphabet.index(s)
                XE_matrix[row][col] += 1
        return XE_matrix  # /len(top_seqs)

    def sample_matrix(self, pwm):
        """Sample."""
        out_seq = ""
        for col in range(pwm.shape[1]):
            sample = np.random.uniform()
            index = min(
                bisect.bisect_left(np.cumsum(pwm[:, col]), sample),
                len(self.alphabet) - 1,
            )
            out_seq += self.alphabet[index]
        return out_seq

    def generate_sequences(self):
        """Generate."""
        offspring = []
        seq_and_fitness = []
        last_batch = self.get_last_batch()
        current_population = self.batches[last_batch]

        for seq in set(current_population):
            seq_and_fitness.append((self.model.get_fitness(seq), seq))

        top_seqs_and_fits = sorted(seq_and_fitness, reverse=True)

        top_f = top_seqs_and_fits[0][0]
        top_seqs = []
        for f, seq in top_seqs_and_fits:
            if f >= top_f * (1 - self.threshold):
                top_seqs.append(seq)
            else:
                break

        XE_matrix_base = self.get_matrix(top_seqs)
        inv_beta = self.beta
        attempt_count = 1
        while (
            len(offspring) < self.batch_size * self.virtual_screen
            and attempt_count < self.virtual_screen
        ):
            added = False
            XE_matrix = softmax(XE_matrix_base, 1.0 / self.beta)
            new_seq = self.sample_matrix(XE_matrix)
            if (
                new_seq not in offspring
                and new_seq not in self.model.measured_sequences
            ):
                offspring.append(new_seq)
                added = True
                attempt_count = 1

            inv_beta *= attempt_count

            if not added:
                attempt_count += 1

        seq_and_fitness = []
        for seq in set(offspring):
            seq_and_fitness.append((self.model.get_fitness(seq), seq))

        if len(seq_and_fitness) == 0:
            seq_and_fitness = top_seqs_and_fits

        return sorted(seq_and_fitness, reverse=True)

    def propose_samples(self):
        """Propose new samples for production"""
        new_seqs_and_fitnesses = self.generate_sequences()
        new_batch = new_seqs_and_fitnesses[: self.batch_size]
        batch_seq = []

        for _, seq in new_batch:
            batch_seq.append(seq)

        return batch_seq
