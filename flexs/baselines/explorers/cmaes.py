"""CMAES explorer."""
from typing import Optional, Tuple

import cma
import numpy as np
import pandas as pd

import flexs
from flexs.utils import sequence_utils as s_utils


class CMAES(flexs.Explorer):
    """
    An explorer which implements the covariance matrix adaptation evolution
    strategy (CMAES).

    Optimizes a continuous relaxation of the one-hot sequence that we use to
    construct a normal distribution around, sample from, and then argmax to get
    sequences for the objective function.

    http://blog.otoro.net/2017/10/29/visual-evolution-strategies/ is a helpful guide.
    """

    def __init__(
        self,
        model: flexs.Model,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        population_size: int = 15,
        max_iter: int = 400,
        initial_variance: float = 0.2,
        log_file: Optional[str] = None,
    ):
        """
        Args:
            population_size: Number of proposed solutions per iteration.
            max_iter: Maximum number of iterations.
            initial_variance: Initial variance passed into cma.
        """
        name = f"CMAES_popsize{population_size}"

        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        self.alphabet = alphabet
        self.population_size = population_size
        self.max_iter = max_iter
        self.initial_variance = initial_variance
        self.round = 0

    def _soln_to_string(self, soln):
        x = soln.reshape((len(self.starting_sequence), len(self.alphabet)))

        one_hot = np.zeros(x.shape)
        one_hot[np.arange(len(one_hot)), np.argmax(x, axis=1)] = 1

        return s_utils.one_hot_to_string(one_hot, self.alphabet)

    def propose_sequences(
        self, measured_sequences: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        measured_sequence_dict = dict(
            zip(measured_sequences["sequence"], measured_sequences["true_score"])
        )

        # Keep track of new sequences generated this round
        top_idx = measured_sequences["true_score"].argmax()
        top_seq = measured_sequences["sequence"].to_numpy()[top_idx]
        top_val = measured_sequences["true_score"].to_numpy()[top_idx]
        sequences = {top_seq: top_val}

        def objective_function(soln):
            seq = self._soln_to_string(soln)

            if seq in sequences:
                return sequences[seq]
            if seq in measured_sequence_dict:
                return measured_sequence_dict[seq]

            return self.model.get_fitness([seq]).item()

        # Starting solution gives equal weight to all residues at all positions
        x0 = s_utils.string_to_one_hot(top_seq, self.alphabet).flatten()
        opts = {"popsize": self.population_size, "verbose": -9, "verb_log": 0}
        es = cma.CMAEvolutionStrategy(x0, np.sqrt(self.initial_variance), opts)

        # Explore until we reach `self.max_iter` or run out of model queries
        initial_cost = self.model.cost
        for _ in range(self.max_iter):

            # Stop exploring if we will run out of model queries
            current_cost = self.model.cost - initial_cost
            if current_cost + self.population_size > self.model_queries_per_batch:
                break

            # `ask_and_eval` generates a new population of sequences
            solutions, fitnesses = es.ask_and_eval(objective_function)
            # `tell` updates model parameters
            es.tell(solutions, fitnesses)

            # Store scores of generated sequences
            for soln, f in zip(solutions, fitnesses):
                sequences[self._soln_to_string(soln)] = f

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        # Negate `objective_function` scores
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        return new_seqs[sorted_order], preds[sorted_order]
