"""Elitist explorers."""
import bisect
import random

import numpy as np

import flsd
import flsd.utils.sequence_utils as s_utils


class Adalead(flsd.Explorer):
    """
    ADALEAD explorer.
    """

    def __init__(
        self,
        model,
        landscape,
        rounds,
        experiment_budget,
        query_budget,
        initial_sequences,

        alphabet,

        mu=1,
        recomb_rate=0,
        threshold=0.05,
        batch_size=100,
        virtual_screen=10,
        rho=1,
    ):
        """Greedy algorithm implementation."""
        super().__init__(model, landscape, rounds, experiment_budget, query_budget, initial_sequences)
        self.threshold = threshold
        self.recomb_rate = recomb_rate
        self.alphabet = alphabet
        self.mu = mu  # number of mutations per *sequence*.
        self.rho = rho
        self.explorer_type = (
            f"Greedy_mu{self.mu}_tr{self.threshold}_r{self.recomb_rate}_rho{self.rho}"
        )

    def _recombine_population(self, gen):
        random.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
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

    def propose_sequences(self, measured_sequences):
        """Generate."""

        measured_sequence_dict = dict(
            zip(measured_sequences['sequence'], measured_sequences['ground_truth'])
        )

        top_fitness = measured_sequences['ground_truth'].max()
        top_inds = measured_sequences['ground_truth'] \
                    >= top_fitness * (1 - np.sign(top_fitness) * self.threshold)

        parents = np.resize(measured_sequences['sequence'][top_inds].to_numpy(), self.experiment_budget)

        sequences = {}
        while len(sequences) < self.query_budget:
            # generate recombinant mutants
            for i in range(self.rho):
                parents = self._recombine_population(parents)

            for root in parents:
                # Here we do rollots from each parent (root of rollout tree)
                root_fitness = self.model.get_fitness([root]).item()
                node = root

                while len(sequences) < self.query_budget:
                    child = s_utils.generate_random_mutant(
                        node, self.mu * 1 / len(node), self.alphabet
                    )

                    # Skip if child has been already been generated before
                    if child in measured_sequence_dict or child in sequences:
                        continue

                    # Stop the rollout once the child has worse predicted
                    # fitness than the root of the rollout tree.
                    # Otherwise, set node = child and add child to the list
                    # of sequences to propose. 
                    child_fitness = self.model.get_fitness([child]).item()
                    sequences[child] = child_fitness

                    if child_fitness >= root_fitness:
                        node = child
                    else:
                        break

        # We propose the top `self.experiment_budget` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[:-self.experiment_budget:-1]

        return new_seqs[sorted_order], preds[sorted_order]