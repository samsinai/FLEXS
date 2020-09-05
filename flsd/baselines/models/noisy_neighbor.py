import editdistance
import numpy as np
import random
from sklearn.metrics import explained_variance_score, r2_score
from scipy.stats import pearsonr

import flsd


class NoisyNeighbor(flsd.Model):
    """
    Behaves like a ground truth model.

    It corrupts a ground truth model with noise, which is modulated by distance
    to already measured sequences.
    """

    def __init__(
        self,
        landscape,
        signal_strength=0.9,
        cache=True,
        landscape_id=-1,
        start_id=-1,
    ):
        self.landscape = landscape
        self.measured_sequences = {}  # save the measured sequences for the model
        self.model_sequences = {}  # cache the sequences for later queries
        self.cost = 0
        self.evals = 0
        self.ss = signal_strength
        self.cache = cache
        self.model_type = f"NAMb_ss{self.ss}"
        self.landscape_id = landscape_id
        self.start_id = start_id
        self.fitnesses = []
        self.r2 = signal_strength ** 2  # this is a proxy

    def get_min_distance(self, sequence):
        new_dist = 1000
        closest = None

        for seq in self.measured_sequences:
            dist = editdistance.eval(sequence, seq)

            if dist == 1:
                return dist, seq

            if dist < new_dist:
                new_dist = dist
                closest = seq

        return new_dist, closest

    def add_noise(self, sequence, distance, neighbor_seq):
        signal = self.landscape.get_fitness([sequence])

        neighbor_seq_fitness = self.landscape.get_fitness([neighbor_seq])
        noise = np.random.exponential(scale=neighbor_seq_fitness)

        alpha = (self.ss) ** distance
        return signal, noise, alpha

    def _fitness_function(self, sequence):
        if self.ss < 1:
            distance, neighbor_seq = self.get_min_distance(sequence)
            signal, noise, alpha = self.add_noise(sequence, distance, neighbor_seq)
            surrogate_fitness = signal * alpha + noise * (1 - alpha)

        else:
            surrogate_fitness = self.landscape.get_fitness([sequence])

        return surrogate_fitness

    def measure_true_landscape(self, sequences):
        predictions = []
        results = []
        for sequence in sequences:
            if sequence not in self.measured_sequences:
                self.cost += 1
                fitness = self.landscape.get_fitness(sequence)
                if sequence in self.model_sequences:
                    model_fitness = self.model_sequences[sequence]
                    predictions.append(model_fitness)
                    results.append(fitness)
                self.measured_sequences[sequence] = fitness
                self.fitnesses.append(fitness)
        if results:
            # self.r2 =r2_score(results,predictions)
            try:
                self.r2 = pearsonr(results, predictions)[0] ** 2
            except:
                pass
        self.model_sequences = {}  # empty cache

    def train(self, sequences, labels):
        self.measured_sequences.update(zip(sequences, labels))

    def get_fitness(self, sequences):
        return np.concatenate([self._fitness_function(seq) for seq in sequences])

    def get_fitness_distribution(self, sequence):

        if sequence in self.measured_sequences:
            real_fitness = self.measured_sequences[sequence]
            return [real_fitness for i in range(5)]
        else:
            estimated_fitnesses = [self._fitness_function(sequence) for i in range(5)]
            self.evals += 1
            return estimated_fitnesses


