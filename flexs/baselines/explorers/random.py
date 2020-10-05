"""Defines the Random explorer class."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import flexs
from flexs.utils import sequence_utils as s_utils


class Random(flexs.Explorer):
    """A simple random explorer.

    Chooses a random previously measured sequence and mutates it.

    A good baseline to compare other search strategies against.

    Since random search is not data-driven, the model is only used to score
    sequences, but not to guide the search strategy.
    """

    def __init__(
        self,
        model: flexs.Model,
        rounds: int,
        starting_sequence: str,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        alphabet: str,
        mu: float = 1,
        elitist: bool = False,
        seed: Optional[int] = None,
        log_file: Optional[str] = None,
    ):
        """
        Create a random search explorer.

        Args:
            mu: Average number of residue mutations from parent for generated sequences.
            elitist: If true, will propose the top `sequences_batch_size` sequences
                generated according to `model`. If false, randomly proposes
                `sequences_batch_size` sequences without taking model score into
                account (true random search).
            seed: Integer seed for random number generator.

        """
        name = f"Random_mu={mu}"

        super().__init__(
            model,
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
        self.elitist = elitist

    def propose_sequences(
        self, measured_sequences: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
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

        if self.elitist:
            idxs = np.argsort(preds)[: -self.sequences_batch_size : -1]
        else:
            idxs = self.rng.integers(0, len(new_seqs), size=self.sequences_batch_size)

        return new_seqs[idxs], preds[idxs]
