import tensorflow
import numpy as np
from meta.model import Model
import random
from utils.sequence_utils import translate_string_to_one_hot
from sklearn.metrics import explained_variance_score, r2_score
from scipy.stats import pearsonr
from tensorflow import keras
import sklearn

from utils.model_architectures import SKGP

# TODO: These names need to get fixed to reflect that it supports SKlearn models as well
class NN_model(Model):
    def __init__(
        self,
        ground_truth_oracle,
        nn_architecture,
        cache=True,
        batch_update=False,
        landscape_id=-1,
        start_id=-1,
    ):
        self.measured_sequences = {}  # save the measured sequences for the model
        self.model_sequences = {}  # cache the sequences for later queries
        self.cost = 0
        self.evals = 0
        self.r2 = 0
        self.cache = cache
        self.landscape_id = landscape_id
        self.start_id = start_id
        self.oracle = ground_truth_oracle
        self.batch_update = batch_update
        self.neuralmodel = nn_architecture.get_model()
        self.batch_size = nn_architecture.batch_size
        self.validation_split = nn_architecture.validation_split
        self.epochs = nn_architecture.epochs
        self.alphabet = nn_architecture.alphabet
        self.nn_architecture = nn_architecture
        self.one_hot_sequences = {}
        self.model_type = f"arch={nn_architecture.architecture_name}"
        if "NN" in self.model_type:
            self.model_flavor = "Keras"
        else:
            self.model_flavor = "SKL"

    def reset(self, sequences=None):
        self.model_sequences = {}
        self.measured_sequences = {}
        self.cost = 0
        self.r2 = 0
        if self.model_flavor == "Keras":
            self.neuralmodel = tensorflow.keras.models.clone_model(self.neuralmodel)
            self.neuralmodel.compile(
                loss="mean_squared_error", optimizer="adam", metrics=["mse"]
            )
        else:
            self.neuralmodel = sklearn.base.clone(self.neuralmodel)

        if sequences:
            self.update_model(sequences)

    def bootstrap(self, wt, alphabet):
        sequences = [wt]
        self.wt = wt
        self.alphabet = alphabet
        for i in range(len(wt)):
            tmp = list(wt)
            for j in range(len(alphabet)):
                tmp[i] = alphabet[j]
                sequences.append("".join(tmp))
        self.measure_true_landscape(sequences)
        if self.model_flavor == "NN":
            self.one_hot_sequences = {
                sequence: (
                    translate_string_to_one_hot(sequence, self.alphabet),
                    self.measured_sequences[sequence],
                )
                for sequence in sequences
            }  # removed flatten for nn
        else:
            self.one_hot_sequences = {
                sequence: (
                    translate_string_to_one_hot(sequence, self.alphabet).flatten(),
                    self.measured_sequences[sequence],
                )
                for sequence in sequences
            }

        self.update_model(sequences)

    def update_model(self, sequences):
        X = []
        Y = []
        self.measure_true_landscape(sequences)
        for sequence in sequences:
            if sequence not in self.one_hot_sequences:
                if self.model_flavor == "Keras":
                    x = translate_string_to_one_hot(
                        sequence, self.alphabet
                    )
                else:
                    x = translate_string_to_one_hot(sequence, self.alphabet).flatten()
                y = self.measured_sequences[sequence]
                self.one_hot_sequences[sequence] = (x, y)
                X.append(x)
                Y.append(y)
            else:
                x, y = self.one_hot_sequences[sequence]
                X.append(x)
                Y.append(y)
        X = np.array(X)
        Y = np.array(Y)

        try:
            y_pred = self.neuralmodel.predict(X)
            if self.model_flavor == "Keras":
                y_pred = y_pred.flatten()

            self.r2 = pearsonr(Y, y_pred)[0] ** 2

        except:
            pass

        if self.model_flavor == "Keras":
            self.neuralmodel.fit(
                X,
                Y,
                epochs=self.epochs,
                validation_split=self.validation_split,
                batch_size=self.batch_size,
                verbose=0,
            )
        else:
            self.neuralmodel.fit(X, Y)

        if not self.batch_update:
            self.retrain_model()

    def retrain_model(self):
        X, Y = [], []
        random_sequences = random.sample(
            self.one_hot_sequences.keys(),
            min(len(self.one_hot_sequences.keys()), self.batch_size * 5),
        )
        for sequence in random_sequences:
            x, y = self.one_hot_sequences[sequence]
            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)

        if self.model_flavor == "Keras":
            self.neuralmodel.fit(
                X,
                Y,
                epochs=self.epochs,
                validation_split=self.validation_split,
                batch_size=self.batch_size,
                verbose=0,
            )
        else:
            self.neuralmodel.fit(X, Y)

    def _fitness_function(self, sequence):
        try:
            if self.model_flavor == "Keras":
                x = np.array([translate_string_to_one_hot(sequence, self.alphabet)])
                return max(min(1, self.neuralmodel.predict(x)[0][0]), 0)

            else:
                x = np.array(
                    [translate_string_to_one_hot(sequence, self.alphabet).flatten()]
                )
                return max(min(1, self.neuralmodel.predict(x)[0]), 0)

        except:
            print(sequence)

    def measure_true_landscape(self, sequences):
        for sequence in sequences:
            if sequence not in self.measured_sequences:
                self.cost += 1
                self.measured_sequences[sequence] = self.oracle.get_fitness(sequence)

        self.model_sequences = {}  # empty cache

    def get_fitness(self, sequence):

        if sequence in self.measured_sequences:
            return self.measured_sequences[sequence]
        elif (
            sequence in self.model_sequences and self.cache
        ):  # caching model answer to save computation
            return self.model_sequences[sequence]

        else:
            self.model_sequences[sequence] = self._fitness_function(sequence)
            self.evals += 1
            return self.model_sequences[sequence]

class GPR_model(NN_model):
    def __init__(
        self,
        ground_truth_oracle,
        nn_architecture,
        cache=True,
        batch_update=False,
        landscape_id=-1,
        start_id=-1,
    ):
        super().__init__(ground_truth_oracle=ground_truth_oracle,
            nn_architecture=nn_architecture,
            cache=cache,
            batch_update=batch_update,
            landscape_id=landscape_id,
            start_id=start_id)

    def _one_hot_to_num(self, one_hot):
        return np.argmax(np.transpose(one_hot), axis=1)

    def update_model(self, sequences):
        X = []
        Y = []
        self.measure_true_landscape(sequences)
        for sequence in sequences:
            if sequence not in self.one_hot_sequences:
                x = translate_string_to_one_hot(sequence, self.alphabet)
                x = self._one_hot_to_num(x)
                y = self.measured_sequences[sequence]
                self.one_hot_sequences[sequence] = (x, y)
                if len(X) == 0:
                    X = [x]
                else:
                    X = np.vstack((X, x))
                Y.append(y)
            else:
                x, y = self.one_hot_sequences[sequence]
                if len(X) == 0:
                    X = [x]
                else:
                    X = np.vstack((X, x))
                Y.append(y)
        X = np.array(X)
        Y = np.array(Y)

        self.neuralmodel.fit(X, Y)

    def _fitness_function(self, sequence, return_std=True):
        # try:
        x = translate_string_to_one_hot(sequence, self.alphabet)
        x = self._one_hot_to_num(x)
        return self.neuralmodel.predict(x.reshape(1, -1), return_std=return_std)
        # except:
        #     print(sequence)

    def get_fitness(self, sequence, return_std=True):
        if sequence in self.measured_sequences:
            if return_std:
                return self.measured_sequences[sequence], 0
            else:
                return self.measured_sequences[sequence]
        elif (
            sequence in self.model_sequences and self.cache
        ):  # caching model answer to save computation
            return self.model_sequences[sequence]

        else:
            self.model_sequences[sequence] = self._fitness_function(sequence, return_std=return_std)
            self.evals += 1
            return self.model_sequences[sequence]