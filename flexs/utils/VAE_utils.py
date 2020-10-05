"""Utility functions for A VAE generative model."""
import random

import numpy as np
import scipy.special
import tensorflow as tf
from tensorflow import keras

from flexs.types import SEQUENCES_TYPE
from flexs.utils import sequence_utils as s_utils


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a sequence."""

    def call(self, inputs):
        """Sample from multivariate guassian defined by
        `inputs = (z_mean, z_log_var)`.

        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEModel(keras.Model):
    """Keras implementation of VAE for CbAS/DbAS."""

    def __init__(
        self, original_dim: int, intermediate_dim: int, latent_dim: int, **kwargs
    ):
        """Create the VAE."""
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
        """Return the VAE's reconstruction of `data`."""
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        return reconstruction

    def generate(self):
        """Generate a new sequence by sampling the latent space and then decoding."""
        z = np.random.randn(1, self.latent_dim)
        return self.decoder(z)

    def train_step(self, data):
        """Define a custom train step taking in `data` and returning the loss."""
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
    """VAE class wrapping `VAEModel`, exposing an interface friendly to CbAS/DbAS."""

    def __init__(
        self,
        seq_length: int,
        alphabet: str,
        batch_size: int = 10,
        latent_dim: int = 2,
        intermediate_dim: int = 250,
        epochs: int = 10,
        epsilon_std: float = 1.0,
        beta: float = 1,
        validation_split: float = 0.2,
        verbose: bool = True,
    ):
        """Create the VAE."""
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
        """Train VAE on `samples` according to their `weights`."""
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
        Generate `n_samples` new samples such that none of them
        are in `existing_samples`.
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

    def calculate_log_probability(
        self, sequences: SEQUENCES_TYPE, vae: VAEModel = None
    ):
        """Calculate log probability of reconstructing a sequence."""
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
    """Convert pwm to boltzmann weights for categorical distribution sampling."""
    weights = np.array(prob_weight_matrix)
    cols_logsumexp = []

    for i in range(weights.shape[1]):
        cols_logsumexp.append(scipy.special.logsumexp(weights.T[i] / temp))

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = np.exp(weights[i, j] / temp - cols_logsumexp[j])

    return weights
