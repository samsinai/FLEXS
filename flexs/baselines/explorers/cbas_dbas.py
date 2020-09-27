"""CbAS and DbAS explorers."""

import random

import numpy as np
import scipy.special
import tensorflow as tf
from tensorflow import keras

import flexs
import flexs.utils.sequence_utils as s_utils


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEModel(keras.Model):
    def __init__(self, original_dim, intermediate_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)

        self.original_dim = original_dim
        self.latent_dim = latent_dim

        # encoding layers
        encoder_inputs = keras.layers.Input(shape=(original_dim))
        x = keras.layers.Dense(intermediate_dim, activation="elu")(encoder_inputs)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(intermediate_dim, activation="elu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(intermediate_dim, activation="elu")(x)
        z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(
            encoder_inputs, [z_mean, z_log_var, z], name="encoder"
        )

        # decoding layers
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = keras.layers.Dense(intermediate_dim, activation="elu")(latent_inputs)
        x = keras.layers.Dense(intermediate_dim, activation="elu")(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(intermediate_dim, activation="elu")(x)
        decoder_outputs = keras.layers.Dense(self.original_dim, activation="sigmoid")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    def call(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        return reconstruction

    def generate(self):
        # sampling from the latent space (normal distribution in this case)
        z = np.random.randn(1, self.latent_dim)
        return self.decoder(z)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = self.original_dim * tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


class VAE:
    def __init__(
        self,
        seq_length,
        alphabet,
        batch_size=10,
        latent_dim=2,
        intermediate_dim=250,
        epochs=10,
        epsilon_std=1.0,
        beta=1,
        validation_split=0.2,
        verbose=True,
    ):
        tf.config.run_functions_eagerly(True)

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epochs = epochs
        self.epsilon_std = epsilon_std
        self.beta = beta
        self.validation_split = validation_split
        self.verbose = verbose
        self.name = f"VAE_latent_dim={latent_dim}_intermediate_dim={intermediate_dim}"

        self.alphabet = alphabet
        self.seq_length = seq_length

        self.vae = VAEModel(
            len(self.alphabet) * self.seq_length, intermediate_dim, latent_dim
        )
        self.vae.compile(optimizer=keras.optimizers.Adam(lr=0.0001, clipvalue=0.5))

    def train_model(self, samples, weights):
        x_train = np.array(
            [s_utils.string_to_one_hot(sample, self.alphabet) for sample in samples],
            dtype="float32",
        )
        x_train = x_train.reshape((len(x_train), self.seq_length * len(self.alphabet)))

        early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=3)

        self.vae.fit(
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

        x_reconstructed_matrix = np.reshape(
            self.vae.generate(), (len(self.alphabet), self.seq_length)
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

    def calculate_log_probability(self, sequences, vae=None):
        if not vae:
            vae = self.vae

        one_hots = np.array(
            [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences],
            dtype="int",
        )
        flattened_one_hots = one_hots.reshape(len(sequences), -1)

        flattened_decoded = vae.predict(flattened_one_hots)
        decoded = flattened_decoded.reshape(
            (len(sequences), self.seq_length, len(self.alphabet))
        )

        # Get the (normalized) probability of reconstructing each
        # particular residue of the original sequence
        per_res_probs = (decoded * one_hots).max(axis=2) / decoded.sum(axis=2)

        # log(prob of reconstructing a sequence)
        #   = log(product of reconstructing each residue in sequence)
        #   = sum(log(product of reconstructing each residue in sequence))
        # We calculate this product as a sum of logs for numerical stability
        log_probs = np.log(1e-9 + per_res_probs).sum(axis=1)

        return np.nan_to_num(log_probs)


def pwm_to_boltzmann_weights(prob_weight_matrix, temp):
    weights = np.array(prob_weight_matrix)
    cols_logsumexp = []

    for i in range(weights.shape[1]):
        cols_logsumexp.append(scipy.special.logsumexp(weights.T[i] / temp))

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = np.exp(weights[i, j] / temp - cols_logsumexp[j])

    return weights


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
        # we will pick all of last round's sequences with fitness above the Qth percentile
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

            # cbas and dbas mostly the same except cbas also does an importance sampling step
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
