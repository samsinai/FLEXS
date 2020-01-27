import sys
# from sklearn.linear_model import LinearRegression,Lasso, LogisticRegression
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D

import keras.backend as K
from keras import objectives
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from utils.sequence_utils import translate_string_to_one_hot, translate_one_hot_to_string
import numpy as np
from scipy.special import logsumexp
import random
from utils.sequence_utils import generate_random_mutant


class Architecture():
    def __init__(self, seq_len, batch_size=10, validation_split=0.1, epochs=20, alphabet="UCGA"):
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.epochs = epochs
        self.alphabet = alphabet
        self.seq_len = seq_len
        self.architecture_name='LNN'

    @property
    def alphabet_len(self):
        return len(self.alphabet)

    def get_model(self):
        raise NotImplementedError( "You need to define an Architecture")


class Linear(Architecture):

    def get_model(self):

        lin_model=Sequential()
        lin_model.add(Flatten())
        lin_model.add(Dense(1,input_shape=(self.seq_len* self.alphabet_len,),use_bias=False))
        lin_model.add(Activation('linear')) 
        lin_model.compile(loss='mean_squared_error', optimizer="adam",metrics=['mse'])  
        return lin_model 

class NLNN(Architecture):
    """Global epistasis model"""
    def __init__(self, seq_len, batch_size=10, validation_split=0.1, epochs=20, alphabet="UCGA", hidden_dims=50):
        super(NLNN, self).__init__(seq_len, batch_size, validation_split, epochs, alphabet)
        self.hidden_dims = hidden_dims
        self.architecture_name=f'NLNN_hd{self.hidden_dims}'


    def get_model(self):

        non_lin_model=Sequential()
        non_lin_model.add(Flatten())
        non_lin_model.add(Dense(1,input_shape=(self.seq_len* self.alphabet_len,),use_bias=False))
        non_lin_model.add(Activation('linear'))
        non_lin_model.add(Dense(self.hidden_dims))
        non_lin_model.add(Activation("relu"))
        non_lin_model.add(Dense(self.hidden_dims))
        non_lin_model.add(Activation("relu"))
        non_lin_model.add(Dense(1))
        non_lin_model.add(Activation("linear"))
        non_lin_model.compile(loss='mean_squared_error', optimizer="adam",metrics=['mse'])  
        return non_lin_model 



class CNNa(Architecture):

    def __init__(self, seq_len, batch_size=10, validation_split=0.1, epochs=20, alphabet="UCGA", filters=50, hidden_dims=100):
        super(CNNa, self).__init__(seq_len, batch_size, validation_split, epochs, alphabet)
        self.filters = filters
        self.hidden_dims = hidden_dims
        self.architecture_name=f'CNNa_hd{self.hidden_dims}_f{self.filters}'


    def get_model(self):
        filters = self.filters 
        hidden_dims = self.hidden_dims

        model = Sequential()
        model.add(Conv1D(filters,
                         self.alphabet_len-1,  
                         padding='valid',
                         strides=1,
                         input_shape=(self.alphabet_len,self.seq_len)))

        model.add(Conv1D(filters,
                         20,
                         padding='same',
                         activation='relu',
                         strides=1))

        model.add(MaxPooling1D(1))
        model.add(Conv1D(filters,
                         self.alphabet_len-1,
                         padding='same',
                         activation='relu',
                         strides=1))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(hidden_dims))
        model.add(Activation('relu'))
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.25))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error',  optimizer="adam", metrics=['mse'])
        return model


class VAE(Architecture):

    def __init__(self, alphabet=None, batch_size=100, latent_dim=2, intermediate_dim=250, epochs=10,\
                 epsilon_std=1.0, beta=1, validation_split=0.05, min_training_size=100, mutation_rate=0.1,
                 verbose=True):
        self.alphabet = alphabet
        self.KEY_LIST = list(alphabet)
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


    def _sampling(self, args):  # reparameterization
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0., stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon


    def _vae_loss(self, x, x_decoded_mean):
        xent_loss = self.original_dim * objectives.categorical_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + self.beta * kl_loss


    def get_model(self, seq_size=0):

        self.seq_size = seq_size
        self.original_dim = len(self.KEY_LIST) * self.seq_size
        self.output_dim = len(self.KEY_LIST) * self.seq_size

        # encoding layers
        x = Input(batch_shape=(self.batch_size, self.original_dim))
        h = Dense(self.intermediate_dim, input_shape=(self.batch_size, self.original_dim), activation="elu")(x)
        h = Dropout(0.7)(h)
        h = Dense(self.intermediate_dim, activation='elu')(h)
        h = BatchNormalization()(h)
        h = Dense(self.intermediate_dim, activation='elu')(h)

        # latent layers
        self.z_mean = Dense(self.latent_dim)(h)
        self.z_log_var = Dense(self.latent_dim)(h)
        z = Lambda(self._sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # decoding layers
        decoder_1 = Dense(self.intermediate_dim, activation='elu')
        decoder_2 = Dense(self.intermediate_dim, activation='elu')
        decoder_2d = Dropout(0.7)
        decoder_3 = Dense(self.intermediate_dim, activation='elu')
        decoder_out = Dense(self.output_dim, activation='sigmoid')  # columnwise softmax?
        x_decoded_mean = decoder_out(decoder_3(decoder_2d(decoder_2(decoder_1(z)))))

        self.vae = Model(x, x_decoded_mean)

        self.vae.compile(optimizer="adam", loss=self._vae_loss, metrics=["categorical_accuracy", "top_k_categorical_accuracy"])

        decoder_input = Input(shape=(self.latent_dim,))
        _x_decoded_mean = decoder_out(decoder_3(decoder_2d(decoder_2(decoder_1(decoder_input)))))
        self.decoder = Model(decoder_input, _x_decoded_mean)

        return


    def train_model(self, samples, weights=[]):

        # generate random seqs around the input seq if the sample size is too small
        if len(samples) < self.min_training_size:
            print('Input batch for the VAE too small, generating more sequences...')
            random_mutants = []
            for sample in samples:
                random_mutants.extend(list(set([generate_random_mutant(sample,
                                                                  self.mutation_rate,
                                                                  alphabet=self.alphabet)
                                                for i in range(self.min_training_size*100)])))
            new_samples = random.sample(random_mutants, (self.min_training_size - len(samples)))
            samples.extend(new_samples)
            weights.extend(np.ones(len(new_samples)))


        compatible_len = (len(samples)//self.batch_size)*self.batch_size
        samples = samples[:compatible_len]
        if len(weights) == 0:
            weights = np.ones(compatible_len)
        else:
            weights = weights[:compatible_len]
        print('Training the VAE...')
        x_train = np.array([translate_string_to_one_hot(sample, self.KEY_LIST) for sample in samples])
        x_train = x_train.astype('float32')
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

        early_stop = EarlyStopping(monitor='val_loss', patience=3)

        self.vae.fit(x_train, x_train,
                     verbose=self.verbose,
                     sample_weight=np.array(weights),
                     shuffle=True,
                     epochs=self.epochs,
                     batch_size=self.batch_size,
                     validation_split=self.validation_split, callbacks=[early_stop])
        return


    def generate(self, n_samples, existing_samples):
        """
        generate n_samples new samples such that none of them are in existing_samples
        """
        z = np.random.randn(1, self.latent_dim)  # sampling from the latent space (normal distr in this case)
        x_reconstructed = self.decoder.predict(z)  # decoding
        x_one_hot = np.reshape(x_reconstructed, (len(self.KEY_LIST), self.seq_size))

        # sample from the reconstructed pwm with Boltzmann weights
        # reject repeated sequences and ones that are in existing_samples
        #temp_list = [0.001, 0.01, 0.05, 0.1, 0.2]
        proposals = []
        temperature = 0.001
        subsequent_duplicates = 0
        while len(proposals) < n_samples:
            weights = pwm_to_boltzmann_weights(x_one_hot, temperature)
            new_seq = []
            for pos in range(self.seq_size):
                new_seq.extend(random.choices(self.KEY_LIST, weights[:, pos]))
            new_seq = ''.join(new_seq)
            if (new_seq not in proposals) and (new_seq not in existing_samples):
                proposals.append(new_seq)
            else:
                subsequent_duplicates += 1
            if subsequent_duplicates >= 10:
                temperature = 2*temperature


        return proposals


    def calculate_log_probability(self, proposals):
        probabilities = []
        for sequence in proposals:
            sequence_one_hot = np.array(translate_string_to_one_hot(sequence, self.KEY_LIST))
            sequence_one_hot_flattened = sequence_one_hot.flatten()
            sequence_one_hot_flattened_batch = np.array([sequence_one_hot_flattened for i in range(self.batch_size)])
            sequence_decoded_flattened = self.vae.predict(sequence_one_hot_flattened_batch, batch_size=self.batch_size)
            sequence_decoded = np.reshape(sequence_decoded_flattened, (self.batch_size, len(self.KEY_LIST), self.seq_size))[0]
            #log_prob = np.trace(np.log(np.matmul(sequence_one_hot.T,sequence_decoded)))
            log_prob = np.sum(np.log(np.sum(sequence_one_hot*sequence_decoded,axis=0)))
            probabilities.append(log_prob)
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





  




