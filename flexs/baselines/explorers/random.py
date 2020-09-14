import numpy as np
import pandas as pd

import flexs
from flexs.utils import sequence_utils as s_utils

from typing import Tuple


class Random(flexs.Explorer):
    """
    A simple random explorer. Chooses a random previously measured sequence
    and mutates it.

    A good baseline to compare other search strategies against.

    Since random search is not data-driven, the model is only used to score
    sequences, but not to guide the search strategy.

    Args:
        model: The model to score generated sequences with.
        landscape: The ground truth landscape.

    """

    def __init__(
        self,
        model: flexs.Model,
        landscape: flexs.Landscape,
        rounds: int,
        mu: float,
        starting_sequence: str,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        alphabet: str,
        log_file: str = None,
        seed: int = None,
    ):
        name = f"Random_mu={mu}"

        super().__init__(
            model,
            landscape,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )
        self.mu = mu
        self.rng = np.random.default_rng(seed)
        self.alphabet = alphabet
        self.name = f"Random_mu{self.mu}"

    def propose_sequences(
        self, measured_sequences: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:

        """Propose `sequences_batch_size` samples."""

        old_sequences = measured_sequences["sequence"]
        old_sequence_set = set(old_sequences)
        new_seqs = set()

        while len(new_seqs) <= self.model_queries_per_batch:
            seq = self.rng.choice(old_sequences)
            new_seq = s_utils.generate_random_mutant(
                seq, self.mu / len(seq), alphabet=self.alphabet
            )

            if new_seq not in old_sequence_set:
                new_seqs.add(new_seq)

        new_seqs = np.array(list(new_seqs))
        preds = self.model.get_fitness(new_seqs)
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        return new_seqs[sorted_order], preds[sorted_order]
