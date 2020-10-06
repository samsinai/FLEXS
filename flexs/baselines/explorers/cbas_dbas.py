"""CbAS and DbAS explorers."""
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import flexs
from flexs.utils import sequence_utils as s_utils
from flexs.utils.VAE_utils import VAE


class CbAS(flexs.Explorer):
    """CbAS and DbAS explorers."""

    def __init__(
        self,
        model: flexs.Model,
        generator: VAE,
        rounds: int,
        starting_sequence: str,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        alphabet: str,
        algo: str = "cbas",
        Q: float = 0.7,
        cycle_batch_size: int = 100,
        mutation_rate: float = 0.2,
        log_file: Optional[str] = None,
    ):
        """
        Explorer which implements Conditioning by Adaptive Sampling (CbAS)
        and DbAS.

        Paper: https://arxiv.org/pdf/1901.10060.pdf

        Args:
            generator: VAE generator.
            algo (either 'cbas' or 'dbas'): Selects either CbAS or DbAS as the main
                algorithm.
            Q: Percentile used as fitness threshold.
            cycle_batch_size: Number of sequences to propose per cycle.
            mutation_rate: Probability of mutation per residue.

        """
        name = f"{algo}_Q={Q}_generator={generator.name}"
        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        if algo not in ["cbas", "dbas"]:
            raise ValueError("`algo` must be one of 'cbas' or 'dbas'")
        self.algo = algo

        self.generator = generator
        self.alphabet = alphabet
        self.Q = Q  # percentile used as the fitness threshold
        self.cycle_batch_size = cycle_batch_size
        self.mutation_rate = mutation_rate

    def _extend_samples(self, samples, weights):
        # generate random seqs around the input seq if the sample size is too small
        samples = list(samples)
        weights = list(weights)
        sequences = set(samples)
        while len(sequences) < 100:
            sample = random.choice(samples)
            sample = s_utils.generate_random_mutant(
                sample, self.mutation_rate, alphabet=self.alphabet
            )

            if sample not in sequences:
                samples.append(sample)
                weights.append(1)
                sequences.add(sample)

        return np.array(samples), np.array(weights)

    def propose_sequences(
        self, measured_sequences_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        # If we are on the first round, our model has no data yet, so the
        # best policy is to propose random sequences in a small neighborhood.
        last_round = measured_sequences_data["round"].max()
        if last_round == 0:
            sequences = set()
            while len(sequences) < self.sequences_batch_size:
                sequences.add(
                    s_utils.generate_random_mutant(
                        self.starting_sequence,
                        2 / len(self.starting_sequence),
                        self.alphabet,
                    )
                )

            sequences = np.array(list(sequences))
            return sequences, self.model.get_fitness(sequences)

        last_round_sequences = measured_sequences_data[
            measured_sequences_data["round"] == last_round
        ]

        # gamma is our threshold (the self.Q-th percentile of sequences from last round)
        # we will pick all of last round's sequences with fitness above the Qth
        # percentile
        gamma = np.percentile(last_round_sequences["true_score"], 100 * self.Q)
        initial_batch = last_round_sequences["sequence"][
            last_round_sequences["true_score"] >= gamma
        ].to_numpy()
        initial_weights = np.ones(len(initial_batch))

        initial_batch, initial_weights = self._extend_samples(
            initial_batch, initial_weights
        )
        all_samples_and_weights = tuple((initial_batch, initial_weights))

        # this will be the current state of the generator
        self.generator.train_model(initial_batch, initial_weights)

        # save the weights of the initial vae and save it as vae_0:
        # there are issues with keras model saving and loading,
        # so we have to recompile it
        generator_0 = VAE(
            seq_length=self.generator.seq_length,
            alphabet=self.generator.alphabet,
            batch_size=self.generator.batch_size,
            latent_dim=self.generator.latent_dim,
            intermediate_dim=self.generator.intermediate_dim,
            epochs=self.generator.epochs,
            epsilon_std=self.generator.epsilon_std,
            beta=self.generator.beta,
            validation_split=self.generator.validation_split,
            verbose=self.generator.verbose,
        )
        original_weights = self.generator.vae.get_weights()
        generator_0.vae.set_weights(original_weights)
        vae_0 = generator_0.vae

        sequences = {}
        previous_model_cost = self.model.cost
        while self.model.cost - previous_model_cost < self.model_queries_per_batch:
            # generate new samples using the generator (second argument is a list of all
            # existing measured and proposed seqs)
            proposals = []
            proposals = self.generator.generate(
                self.cycle_batch_size,
                all_samples_and_weights[0],
                all_samples_and_weights[1],
            )
            print(self.model.cost - previous_model_cost, len(proposals))

            # calculate the scores of the new samples using the model
            scores = self.model.get_fitness(proposals)

            # set a new fitness threshold if the new percentile is
            # higher than the current
            gamma = max(np.percentile(scores, self.Q * 100), gamma)

            # cbas and dbas mostly the same except cbas also does an importance
            # sampling step
            if self.algo == "cbas":
                # calculate the weights for the proposed batch
                log_probs_0 = self.generator.calculate_log_probability(
                    proposals, vae=vae_0
                )
                log_probs_t = self.generator.calculate_log_probability(proposals)

                weights = np.exp(log_probs_0 - log_probs_t)
                weights = np.nan_to_num(weights)

            # Otherwise, `self.algo == "dbas"`
            else:
                weights = np.ones(len(proposals))

            weights[scores < gamma] = 0

            # add proposed samples to the total sample pool
            all_samples = np.append(all_samples_and_weights[0], proposals)
            all_weights = np.append(all_samples_and_weights[1], weights)
            all_samples_and_weights = (all_samples, all_weights)

            # update the generator
            # print('New training set size: ', len(all_samples_and_weights[0]))
            self.generator.train_model(
                all_samples_and_weights[0], all_samples_and_weights[1]
            )

            sequences.update(zip(proposals, scores))

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        return new_seqs[sorted_order], preds[sorted_order]
