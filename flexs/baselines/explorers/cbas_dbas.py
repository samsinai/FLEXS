"""CbAS and DbAS explorers."""
import random

import numpy as np
from flex.utils.VAE_utils import VAE

import flexs
from flexs.utils import sequence_utils as s_utils


class CbAS(flexs.Explorer):
    """CbAS explorer."""

    def __init__(
        self,
        model,
        generator,
        rounds,
        starting_sequence,
        sequences_batch_size,
        model_queries_per_batch,
        alphabet,
        algo="cbas",
        Q=0.9,
        max_cycles_per_batch=30,
        backfill=True,
        mutation_rate=0.2,
        log_file=None,
    ):
        """Explorer which implements Conditioning by Adaptive Sampling (CbAS).

        Paper: https://arxiv.org/pdf/1901.10060.pdf

        Attributes:
            generator:
            Q:
            n_new_proposals:
            backfill:
            mutation_rate:
            all_proposals_ranked:
            n_convergence: Assume convergence if max fitness doesn't change for
                n_convergence cycles.
            explorer_type:
        """
        name = f"CbAS_Q={Q}_generator={generator.name}"
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
        self.max_cycles_per_batch = max_cycles_per_batch
        self.backfill = backfill
        self.mutation_rate = mutation_rate

    def extend_samples(self, samples, weights):
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

    def propose_sequences(self, measured_sequences_data):
        """Propose `batch_size` samples."""

        last_round_sequences = measured_sequences_data[
            measured_sequences_data["round"] == measured_sequences_data["round"].max()
        ]

        # gamma is our threshold (the self.Q-th percentile of sequences from last round)
        # we will pick all of last round's sequences with fitness above the Qth
        # percentile
        gamma = np.percentile(last_round_sequences["true_score"], 100 * self.Q)
        initial_batch = last_round_sequences["sequence"][
            last_round_sequences["true_score"] >= gamma
        ].to_numpy()
        initial_weights = np.ones(len(initial_batch))

        initial_batch, initial_weights = self.extend_samples(
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

        count = 0  # total count of proposed sequences

        sequences = {}
        current_cycle = 0
        while current_cycle < self.max_cycles_per_batch and (
            count < self.model_queries_per_batch
        ):
            # generate new samples using the generator (second argument is a list of all
            # existing measured and proposed seqs)
            proposals = []
            proposals = self.generator.generate(
                self.sequences_batch_size,
                all_samples_and_weights[0],
                all_samples_and_weights[1],
            )
            count += len(proposals)

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
