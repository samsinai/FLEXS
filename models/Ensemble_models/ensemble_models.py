import editdistance
import numpy as np
from meta.model import Model
import random
from utils.sequence_utils import translate_string_to_one_hot
from utils.model_architectures import keras_architectures, sklearn_architectures
from sklearn.metrics import explained_variance_score, r2_score
import tensorflow
from tensorflow import keras

class EnsembleModel(Model):
    def __init__(
        self, ground_truth_oracle, architectures, model_score_threshold=0.5, cache=True
    ):
        """
        Ensemble Model used for model-based policy training.
        
        Args:
            ground_truth_oracle: Landscape.
            architectures: Array of Architecture objects (from utils.model_architectures).
            model_score_threshold: float value for filtering out poorly scoring models.
        """
        self.measured_sequences = {}  # save the measured sequences for the model
        self.model_sequences = {}  # cache the sequences for later queries
        self.cost = 0
        self.evals = 0

        self.architectures = architectures
        self.models = [arch.get_model() for arch in architectures]
        self.model_r2s = np.zeros(len(self.models))
        self.oracle = ground_truth_oracle
        self.threshold = model_score_threshold

        alphabets = [arch.alphabet for arch in self.architectures]
        alphabets = np.unique(alphabets)
        if len(alphabets) != 1:
            raise ValueError("Must have 1 alphabet for all candidate models.")
        self.alphabet = alphabets[0]

        self.cache = cache
        self.one_hot_sequences = {}

    def reset(self):
        self.model_sequences = {}
        self.measured_sequences = {}
        self.cost = 0

        for i, model in enumerate(self.models):
            self.models[i] = keras.models.clone_model(model)
            self.models[i].compile(
                loss="mean_squared_error", optimizer="adam", metrics=["mse"]
            )

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
        self.one_hot_sequences = {
            sequence: (
                translate_string_to_one_hot(sequence, self.alphabet),
                self.measured_sequences[sequence],
            )
            for sequence in sequences
        }  # removed flatten for nn
        self.update_model(sequences)

    def get_one_hots_and_fitnesses(self, sequences):
        X = []
        Y = []

        for sequence in sequences:
            if sequence not in self.one_hot_sequences:  # or self.batch_update:
                x = translate_string_to_one_hot(sequence, self.alphabet)
                y = self.measured_sequences[sequence]
                self.one_hot_sequences[sequence] = (x, y)
                X.append(x)
                Y.append(y)
            else:
                x, y = self.one_hot_sequences[sequence]
                X.append(x)
                Y.append(y)

        return np.array(X), np.array(Y)

    def fit_candidate_models(self, sequences):
        self.measure_true_landscape(sequences)

        X, Y = self.get_one_hots_and_fitnesses(sequences)

        for i, (model, arch) in enumerate(zip(self.models, self.architectures)):
            if type(arch) in keras_architectures:
                model.fit(
                    X,
                    Y,
                    epochs=arch.epochs,
                    validation_split=arch.validation_split,
                    batch_size=arch.batch_size,
                    verbose=1,
                )
            elif type(arch) in sklearn_architectures:
                X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
                model.fit(X, Y)
            else:
                raise ValueError(type(arch))

    def score_candidate_models(self, sequences):
        X, Y = self.get_one_hots_and_fitnesses(sequences)

        for i, (model, arch) in enumerate(zip(self.models, self.architectures)):
            if type(arch) in keras_architectures:
                y_pred = model.predict(X)
                self.model_r2s[i] = r2_score(Y, y_pred)
            elif type(arch) in sklearn_architectures:
                X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
                y_pred = model.predict(X)
                self.model_r2s[i] = r2_score(Y, y_pred)
            else:
                raise ValueError(type(arch))

    def filter_models(self):
        # filter out models by threshold
        good = self.model_r2s > self.threshold
        print(f"Allowing {np.sum(good)}/{len(good)} models.")
        self.architectures = self.architectures[good]
        self.models = self.models[good]
        self.model_r2s = self.model_r2s[good]

    def _fitness_function(self, sequence):
        reward = 0
        working_models = 0
        for model in self.models:
            try:
                x = np.array([translate_string_to_one_hot(sequence, self.alphabet)])
                reward += max(min(200, model.predict(x)[0][0]), -200)
                working_models += 1
            except:
                print(f"_fitness_function failed on {sequence}")
        return reward / working_models

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
