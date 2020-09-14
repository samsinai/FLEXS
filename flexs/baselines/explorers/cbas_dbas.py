"""CbAS and DbAS explorers."""

import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import flexs
import flexs.utils.sequence_utils as s_utils


class VAE:
    def __init__(
        self,
        seq_length,
        alphabet,
        batch_size=100,
        latent_dim=2,
        intermediate_dim=250,
        epochs=10,
        epsilon_std=1.0,
        beta=1,
        validation_split=0.05,
        min_training_size=100,
        mutation_rate=0.1,
        verbose=True,
    ):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epochs = epochs
        self.epsilon_std = epsilon_std
        self.beta = beta
        self.validation_split = validation_split
        self.min_training_size = min_training_size
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        self.name = "VAE"

        self.alphabet = alphabet
        self.seq_length = seq_length
        self.original_dim = len(self.alphabet) * self.seq_length
        self.output_dim = len(self.alphabet) * self.seq_length

        # encoding layers
        x = keras.layers.Input(batch_shape=(self.batch_size, self.original_dim))
        h = keras.layers.Dense(self.intermediate_dim, activation="elu",)(x)
        h = keras.layers.Dropout(0.7)(h)
        h = keras.layers.Dense(self.intermediate_dim, activation="elu")(h)
        h = keras.layers.BatchNormalization()(h)
        h = keras.layers.Dense(self.intermediate_dim, activation="elu")(h)

        # latent layers
        self.z_mean = keras.layers.Dense(self.latent_dim)(h)
        self.z_log_var = keras.layers.Dense(self.latent_dim)(h)
        z = keras.layers.Lambda(self._sampling, output_shape=(self.latent_dim,))(
            [self.z_mean, self.z_log_var]
        )

        # decoding layers
        decoder_1 = keras.layers.Dense(self.intermediate_dim, activation="elu")
        decoder_2 = keras.layers.Dense(self.intermediate_dim, activation="elu")
        decoder_2d = keras.layers.Dropout(0.7)
        decoder_3 = keras.layers.Dense(self.intermediate_dim, activation="elu")
        decoder_out = keras.layers.Dense(self.output_dim, activation="sigmoid")
        x_decoded_mean = decoder_out(decoder_3(decoder_2d(decoder_2(decoder_1(z)))))

        self.vae = keras.models.Model(x, x_decoded_mean)

        opt = keras.optimizers.Adam(lr=0.0001, clipvalue=0.5)

        self.vae.compile(optimizer=opt, loss=self._vae_loss)

        decoder_input = keras.layers.Input(shape=(self.latent_dim,))
        _x_decoded_mean = decoder_out(
            decoder_3(decoder_2d(decoder_2(decoder_1(decoder_input))))
        )
        self.decoder = keras.models.Model(decoder_input, _x_decoded_mean)

    def _sampling(self, args):  # reparameterization
        z_mean, z_log_var = args
        epsilon = K.random_normal(
            shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=self.epsilon_std
        )
        return z_mean + tf.math.exp(z_log_var / 2) * epsilon

    def _vae_loss(self, x, x_decoded_mean):
        xent_loss = self.original_dim * tf.losses.categorical_crossentropy(
            x, x_decoded_mean
        )
        kl_loss = -0.5 * K.sum(
            1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1
        )
        return xent_loss + self.beta * kl_loss

    def train_model(self, samples, weights):
        # generate random seqs around the input seq if the sample size is too small
        samples = list(samples)
        weights = list(weights)
        sequences = set()
        while len(sequences) < self.min_training_size:
            sample = random.choice(samples)
            sample = s_utils.generate_random_mutant(sample, 3, alphabet=self.alphabet)

            if sample not in sequences:
                samples.append(sample)
                weights.append(1)
                sequences.add(sample)

        x_train = np.array(
            [s_utils.string_to_one_hot(sample, self.alphabet) for sample in samples]
        )
        x_train = x_train.astype("float32")
        x_train = x_train.reshape((len(x_train), self.seq_length * len(self.alphabet)))

        early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=3)

        self.vae.fit(
            x_train,
            x_train,
            verbose=self.verbose,
            sample_weight=np.array(weights),
            shuffle=True,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stop],
        )

    def generate(self, n_samples, existing_samples, existing_weights):
        """
        Generate `n_samples` new samples such that none of them are in existing_samples.
        """
        z = np.random.randn(
            1, self.latent_dim
        )  # sampling from the latent space (normal distribution in this case)
        x_reconstructed = self.decoder.predict(z)  # decoding
        x_reconstructed_matrix = np.reshape(
            x_reconstructed, (len(self.alphabet), self.seq_length)
        )

        if (
            np.isnan(x_reconstructed_matrix).any()
            or np.isinf(x_reconstructed_matrix).any()
        ):
            raise ValueError("NaN and/or inf in the reconstruction matrix")

        # sample from the reconstructed pwm with Boltzmann weights
        # reject repeated sequences and ones that are in existing_samples
        proposals = []
        temperature = 0.001
        weights = pwm_to_boltzmann_weights(x_reconstructed_matrix, temperature)
        repetitions = 0

        while len(proposals) < n_samples:
            new_seq = []
            for pos in range(self.seq_length):
                new_seq.extend(random.choices(self.alphabet, weights[:, pos]))
            new_seq = "".join(new_seq)
            if (new_seq not in proposals) and (new_seq not in existing_samples):
                proposals.append(new_seq)
            else:
                repetitions += 1
                temperature = 1.3 * temperature
                weights = pwm_to_boltzmann_weights(x_reconstructed_matrix, temperature)

        return proposals

    def calculate_log_probability(self, proposals, vae=None):
        if not vae:
            vae = self.vae
        probabilities = []
        for sequence in proposals:
            sequence_one_hot = np.array(
                s_utils.string_to_one_hot(sequence, self.alphabet)
            )
            sequence_one_hot_flattened = sequence_one_hot.flatten()
            sequence_one_hot_flattened_batch = np.array(
                [sequence_one_hot_flattened for i in range(self.batch_size)]
            )
            sequence_decoded_flattened = vae.predict(
                sequence_one_hot_flattened_batch, batch_size=self.batch_size
            )
            sequence_decoded = np.reshape(
                sequence_decoded_flattened,
                (self.batch_size, len(self.alphabet), self.seq_length),
            )[0]
            sequence_decoded = normalize(sequence_decoded, axis=0, norm="l1")
            log_prob = np.sum(
                np.log(10e-10 + np.sum(sequence_one_hot * sequence_decoded, axis=0))
            )
            probabilities.append(log_prob)
        probabilities = np.nan_to_num(probabilities)
        return probabilities


def pwm_to_boltzmann_weights(prob_weight_matrix, temp):
    weights = np.array(prob_weight_matrix)
    cols_logsumexp = []

    for i in range(weights.shape[1]):
        cols_logsumexp.append(logsumexp(weights.T[i] / temp))

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = np.exp(weights[i, j] / temp - cols_logsumexp[j])

    return weights


class CbAS(flexs.Explorer):
    """CbAS explorer."""

    def __init__(
        self,
        model,
        landscape,
        generator,
        rounds,
        starting_sequence,
        sequences_batch_size,
        model_queries_per_batch,
        Q=0.9,
        n_convergence=10,
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
            landscape,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )
        self.generator = generator
        self.Q = Q  # percentile used as the fitness threshold
        self.n_new_proposals = self.sequences_batch_size
        self.backfill = backfill
        self.mutation_rate = mutation_rate
        self.all_proposals_ranked = []
        self.n_convergence = n_convergence

    def propose_sequences(self, measured_sequences):
        """Propose `batch_size` samples."""

        last_round_sequences = measured_sequences[
            measured_sequences["round"] == measured_sequences["round"].max()
        ]

        # gamma is our threshold (the self.Q-th percentile of sequences from last round)
        # we will pick all of last round's sequences with fitness above the Qth percentile
        gamma = np.percentile(last_round_sequences["true_score"], 100 * self.Q)
        initial_batch = last_round_sequences["sequence"][
            last_round_sequences["true_score"] >= gamma
        ].to_numpy()

        initial_weights = np.ones(len(initial_batch))
        all_samples_and_weights = tuple((initial_batch, initial_weights))

        # this will be the current state of the generator
        self.generator.train_model(initial_batch, initial_weights)

        # save the weights of the initial vae and save it as vae_0:
        # there are issues with keras model saving and loading,
        # so we have to recompile it
        self.generator.vae.save("vae_initial_weights.h5")
        generator_0 = VAE(
            seq_length=self.generator.vae.seq_length,
            alphabet=self.generator.vae.alphabet,
            batch_size=self.generator.batch_size,
            latent_dim=self.generator.latent_dim,
            intermediate_dim=self.generator.intermediate_dim,
            epochs=self.generator.epochs,
            epsilon_std=self.generator.epsilon_std,
            beta=self.generator.beta,
            validation_split=self.generator.validation_split,
            min_training_size=self.generator.min_training_size,
            mutation_rate=self.generator.mutation_rate,
            verbose=False,
        )
        generator_0.get_model(
            seq_length=len(self.starting_sequence), alphabet=self.alphabet
        )
        vae_0 = generator_0.vae
        vae_0.load_weights("vae_initial_weights.h5")

        max_fitnesses = (
            []
        )  # keep track of max proposed fitnesses to check for convergence
        not_converged = True
        count = 0  # total count of proposed sequences

        while (not_converged) and (count < self.model_queries_per_batch):

            # generate new samples using the generator (second argument is a list of all
            # existing measured and proposed seqs)
            proposals = []
            while len(proposals) == 0:
                try:
                    proposals = self.generator.generate(
                        self.sequences_batch_size,
                        all_samples_and_weights[0],
                        all_samples_and_weights[1],
                    )
                    # print(f'Proposed {len(proposals)} new samples')
                    count += len(proposals)
                except ValueError as e:
                    print(e.message)
                    print("Ending the CbAS cycle, returning existing proposals...")
                    return self.all_proposals_ranked[-self.n_new_proposals :]

            # calculate the scores of the new samples using the oracle
            scores = []
            for proposal in proposals:
                f = self.model.get_fitness(proposal)
                try:
                    f = f[0]
                except:
                    pass
                scores.append(f)
            # print('Top score in proposed samples: ', np.max(scores))

            # set a new fitness threshold if the new percentile is
            # higher than the current
            gamma_new = np.percentile(scores, self.Q * 100)
            if gamma_new > gamma:
                gamma = gamma_new

            # calculate the weights for the proposed batch
            log_probs_0 = self.generator.calculate_log_probability(proposals, vae=vae_0)
            log_probs_t = self.generator.calculate_log_probability(proposals)
            weights_probs = [
                np.exp(logp0 - logpt)
                for (logp0, logpt) in list(zip(log_probs_0, log_probs_t))
            ]
            weights_probs = np.nan_to_num(weights_probs)
            weights_cdf = [1 if score >= gamma else 0 for score in scores]
            weights = list(np.array(weights_cdf) * np.array(weights_probs))

            # add proposed samples to the total sample pool
            all_samples = all_samples_and_weights[0] + proposals
            all_weights = all_samples_and_weights[1] + weights
            all_samples_and_weights = tuple((all_samples, all_weights))

            # update the generator
            # print('New training set size: ', len(all_samples_and_weights[0]))
            self.generator.train_model(
                all_samples_and_weights[0], all_samples_and_weights[1]
            )

            scores_dict = dict(zip(proposals, scores))
            self.all_proposals_ranked.extend(
                [
                    proposal
                    for proposal, score in sorted(
                        scores_dict.items(), key=lambda item: item[1]
                    )
                ]
            )
            # all_proposals_ranked are in an increasing order or fitness,
            # starting with the first batch

            # check if converged
            max_fitnesses.append(np.max(scores))
            if len(max_fitnesses) >= self.n_convergence:
                if len(set(max_fitnesses[-self.n_convergence :])) == 1:
                    not_converged = False
                    print("CbAS converged")

        self.all_proposals_ranked.reverse()

        if self.backfill:
            return self.all_proposals_ranked[: self.n_new_proposals]

        return [proposal for proposal in proposals if scores_dict[proposal] >= gamma]


'''class DbAS(Base_explorer):
    """DbAS explorer."""

    def __init__(
        self,
        generator=None,
        Q=0.9,
        n_convergence=10,
        backfill=True,
        batch_size=100,
        alphabet="UCGA",
        virtual_screen=10,
        mutation_rate=0.2,
        path="./simulations/",
        debug=False,
    ):
        """Explorer which implements Design by Adaptive Sampling (DbAS).

        Paper: https://arxiv.org/pdf/1810.03714.pdf

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
        super().__init__(
            batch_size, alphabet, virtual_screen, path, debug
        )  # for Python 3
        self.generator = generator
        self.Q = Q  # percentile used as the fitness threshold
        self.n_new_proposals = self.batch_size
        self.backfill = backfill
        self.mutation_rate = mutation_rate
        self.all_proposals_ranked = []
        self.n_convergence = n_convergence
        self.explorer_type = f"DbAS_Q{self.Q}_generator{self.generator.name}"

    def propose_samples(self):
        """Propose `batch_size` samples."""
        gamma = np.percentile(
            list(self.model.measured_sequences.values()), 100 * self.Q
        )  # Qth percentile of current measured sequences
        initial_batch = [
            sequence
            for sequence in self.model.measured_sequences.keys()
            if self.model.measured_sequences[sequence] >= gamma
        ]  # pick all measured sequences with fitness above the Qth percentile
        initial_weights = [1] * len(initial_batch)
        all_samples_and_weights = tuple((initial_batch, initial_weights))

        logging.info("Starting a DbAS cycle...")
        logging.info(f"Initial training set size: {len(initial_batch)}")

        # this will be the current state of the generator
        self.generator.get_model(
            seq_length=len(initial_batch[0]), alphabet=self.alphabet
        )
        self.generator.train_model(initial_batch, initial_weights)

        max_fitnesses = (
            []
        )  # keep track of max proposed fitnesses to check for convergence
        not_converged = True
        count = 0  # total count of proposed sequences

        while (not_converged) and (count < self.batch_size * self.virtual_screen):

            # generate new samples using the generator
            # (second argument is a list of all existing measured and proposed seqs)
            proposals = []
            while len(proposals) == 0:
                try:
                    proposals = self.generator.generate(
                        self.batch_size,
                        all_samples_and_weights[0],
                        all_samples_and_weights[1],
                    )
                    # print(f'Proposed {len(proposals)} new samples')
                    count += len(proposals)
                except ValueError as e:
                    print(e.message)
                    print("Ending the CbAS cycle, returning existing proposals...")
                    return self.all_proposals_ranked[-self.n_new_proposals :]

            # calculate the scores of the new samples using the oracle
            scores = []
            for proposal in proposals:
                f = self.model.get_fitness(proposal)
                try:
                    f = f[0]
                except:
                    pass
                scores.append(f)
            # print('Top score in proposed samples: ', np.max(scores))

            # set a new fitness threshold if the new percentile is
            # higher than the current
            gamma_new = np.percentile(scores, self.Q * 100)
            if gamma_new > gamma:
                gamma = gamma_new

            # calculate the weights for the proposed batch
            weights = [1 if score >= gamma else 0 for score in scores]

            # add proposed samples to the total sample pool
            all_samples = all_samples_and_weights[0] + proposals
            all_weights = all_samples_and_weights[1] + weights
            all_samples_and_weights = tuple((all_samples, all_weights))

            # update the generator
            # print('New training set size: ', len(all_samples_and_weights[0]))
            self.generator.train_model(
                all_samples_and_weights[0], all_samples_and_weights[1]
            )

            scores_dict = dict(zip(proposals, scores))
            self.all_proposals_ranked.extend(
                [
                    proposal
                    for proposal, score in sorted(
                        scores_dict.items(), key=lambda item: item[1]
                    )
                ]
            )
            # all_proposals_ranked are in an increasing order or fitness,
            # starting with the first batch

            # check if converged
            max_fitnesses.append(np.max(scores))
            if len(max_fitnesses) >= self.n_convergence:
                if len(set(max_fitnesses[-self.n_convergence :])) == 1:
                    not_converged = False
                    print("DbAS converged")

        self.all_proposals_ranked.reverse()

        if self.backfill:
            return self.all_proposals_ranked[: self.n_new_proposals]

        return [proposal for proposal in proposals if scores_dict[proposal] >= gamma]
'''
