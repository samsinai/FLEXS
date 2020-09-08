import random
import sys

import numpy as np
import tensorflow
import tensorflow as tf
from scipy.special import logsumexp
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (
    BayesianRidge,
    Lasso,
    LinearRegression,
    LogisticRegression,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import normalize
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalMaxPooling1D,
    Input,
    Lambda,
    MaxPooling1D,
)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from flexs.utils import sequence_utils as s_utils

tf.config.experimental_run_functions_eagerly(True)


class Architecture:
    def __init__(
        self, seq_len, batch_size=10, validation_split=0.0, epochs=20, alphabet="UCGA"
    ):
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.epochs = epochs
        self.alphabet = alphabet
        self.seq_len = seq_len
        self.architecture_name = "LNN"

    @property
    def alphabet_len(self):
        return len(self.alphabet)

    def get_model(self):
        raise NotImplementedError(
            "You must define an Architecture first before you can call get_model."
        )


class SKBR(Architecture):
    def __init__(
        self,
        seq_len,
        batch_size=10,
        validation_split=0.0,
        epochs=20,
        alphabet="UCGA",
        filters=50,
        hidden_dims=100,
    ):
        super(SKBR, self).__init__(
            seq_len, batch_size, validation_split, epochs, alphabet
        )
        self.architecture_name = f"SKBR"

    def get_model(self):
        return BayesianRidge()


class SKLasso(Architecture):
    def __init__(
        self,
        seq_len,
        batch_size=10,
        validation_split=0.0,
        epochs=20,
        alphabet="UCGA",
        filters=50,
        hidden_dims=100,
    ):
        super(SKLasso, self).__init__(
            seq_len, batch_size, validation_split, epochs, alphabet
        )
        self.architecture_name = f"SKLasso"

    def get_model(self):
        return Lasso()


class SKNeighbors(Architecture):
    def __init__(
        self,
        seq_len,
        batch_size=10,
        validation_split=0.0,
        epochs=20,
        alphabet="UCGA",
        filters=50,
        hidden_dims=100,
    ):
        super(SKNeighbors, self).__init__(
            seq_len, batch_size, validation_split, epochs, alphabet
        )
        self.architecture_name = f"SKNeighbors"

    def get_model(self):
        return KNeighborsRegressor()


class SKGB(Architecture):
    def __init__(
        self,
        seq_len,
        batch_size=10,
        validation_split=0.0,
        epochs=20,
        alphabet="UCGA",
        filters=50,
        hidden_dims=100,
    ):
        super(SKGB, self).__init__(
            seq_len, batch_size, validation_split, epochs, alphabet
        )
        self.architecture_name = f"SKGB"

    def get_model(self):
        return GradientBoostingRegressor()


class SKExtraTrees(Architecture):
    def __init__(
        self,
        seq_len,
        batch_size=10,
        validation_split=0.0,
        epochs=20,
        alphabet="UCGA",
        filters=50,
        hidden_dims=100,
    ):
        super(SKExtraTrees, self).__init__(
            seq_len, batch_size, validation_split, epochs, alphabet
        )
        self.architecture_name = f"SKExtraTrees"

    def get_model(self):
        return ExtraTreesRegressor()


class SKGP(Architecture):
    def __init__(
        self,
        seq_len,
        batch_size=10,
        validation_split=0.0,
        epochs=20,
        alphabet="UCGA",
        filters=50,
        hidden_dims=100,
    ):
        super(SKGP, self).__init__(
            seq_len, batch_size, validation_split, epochs, alphabet
        )
        self.architecture_name = f"SKGP"

    def get_model(self):
        return GaussianProcessRegressor()


class VAE(Architecture):
    def __init__(
        self,
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
        self.vae = Model()
        self.decoder = Model()
        self.verbose = verbose
        self.name = "VAE"

    def _sampling(self, args):  # reparameterization
        z_mean, z_log_var = args
        epsilon = K.random_normal(
            shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=self.epsilon_std
        )
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def _vae_loss(self, x, x_decoded_mean):
        xent_loss = self.original_dim * categorical_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(
            1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1
        )
        return xent_loss + self.beta * kl_loss

    def get_model(self, seq_size=0, alphabet=None):
        # This is TF 1
        self.alphabet = alphabet
        self.KEY_LIST = list(self.alphabet)
        self.seq_size = seq_size
        self.original_dim = len(self.KEY_LIST) * self.seq_size
        self.output_dim = len(self.KEY_LIST) * self.seq_size

        # encoding layers
        x = Input(batch_shape=(self.batch_size, self.original_dim))
        h = Dense(
            self.intermediate_dim,
            input_shape=(self.batch_size, self.original_dim),
            activation="elu",
        )(x)
        h = Dropout(0.7)(h)
        h = Dense(self.intermediate_dim, activation="elu")(h)
        h = BatchNormalization()(h)
        h = Dense(self.intermediate_dim, activation="elu")(h)

        # latent layers
        self.z_mean = Dense(self.latent_dim)(h)
        self.z_log_var = Dense(self.latent_dim)(h)
        z = Lambda(self._sampling, output_shape=(self.latent_dim,))(
            [self.z_mean, self.z_log_var]
        )

        # decoding layers
        decoder_1 = Dense(self.intermediate_dim, activation="elu")
        decoder_2 = Dense(self.intermediate_dim, activation="elu")
        decoder_2d = Dropout(0.7)
        decoder_3 = Dense(self.intermediate_dim, activation="elu")
        decoder_out = Dense(self.output_dim, activation="sigmoid")
        x_decoded_mean = decoder_out(decoder_3(decoder_2d(decoder_2(decoder_1(z)))))

        self.vae = Model(x, x_decoded_mean)

        opt = Adam(lr=0.0001, clipvalue=0.5)

        self.vae.compile(optimizer=opt, loss=self._vae_loss)

        decoder_input = Input(shape=(self.latent_dim,))
        _x_decoded_mean = decoder_out(
            decoder_3(decoder_2d(decoder_2(decoder_1(decoder_input))))
        )
        self.decoder = Model(decoder_input, _x_decoded_mean)

        return

    def train_model(self, samples, weights=[]):
        # generate random seqs around the input seq if the sample size is too small
        if len(samples) < self.min_training_size:
            random_mutants = []
            for sample in samples:
                random_mutants.extend(
                    list(
                        set(
                            [
                                s_utils.generate_random_mutant(
                                    sample, self.mutation_rate, alphabet=self.alphabet
                                )
                                for i in range(self.min_training_size * 100)
                            ]
                        )
                    )
                )
            new_samples = random.sample(
                random_mutants, (self.min_training_size - len(samples))
            )
            samples.extend(new_samples)
            weights.extend(np.ones(len(new_samples)))

        compatible_len = (len(samples) // self.batch_size) * self.batch_size
        samples = samples[:compatible_len]
        if len(weights) == 0:
            weights = np.ones(compatible_len)
        else:
            weights = weights[:compatible_len]

        x_train = np.array(
            [s_utils.string_to_one_hot(sample, self.KEY_LIST) for sample in samples]
        )
        x_train = x_train.astype("float32")
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

        early_stop = EarlyStopping(monitor="loss", patience=3)

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

        return

    def generate(self, n_samples, existing_samples, existing_weights):
        """
        Generate `n_samples` new samples such that none of them are in existing_samples.
        """
        z = np.random.randn(
            1, self.latent_dim
        )  # sampling from the latent space (normal distribution in this case)
        x_reconstructed = self.decoder.predict(z)  # decoding
        x_reconstructed_matrix = np.reshape(
            x_reconstructed, (len(self.KEY_LIST), self.seq_size)
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
            for pos in range(self.seq_size):
                new_seq.extend(random.choices(self.KEY_LIST, weights[:, pos]))
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
                s_utils.string_to_one_hot(sequence, self.KEY_LIST)
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
                (self.batch_size, len(self.KEY_LIST), self.seq_size),
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
