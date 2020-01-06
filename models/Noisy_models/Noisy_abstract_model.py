
from meta.model import Model
import editdistance
import numpy as np


class Noisy_abstract_model(Model):
    """Behaves like a ground truth model however corrupts a ground truth model with noise,
    which is modulated by distance to already measured sequences"""

    def __init__(self,ground_truth_oracle, bias=0, max_uncertainty=1, signal_strength=0.5, cache=True, landscape_id=-1, start_id=-1):
        self.oracle = ground_truth_oracle
        self.measured_sequences = {} # save the measured sequences for the model
        self.model_sequences = {} # cache the sequences for later queries
        self.cost = 0
        self.evals = 0
        self._bias = bias
        self.max_uncertainty = max_uncertainty
        self.ss = signal_strength
        self.cache = cache
        self.model_type =f'NAMb{self._bias}ss{self.ss}maxunc{self.max_uncertainty}'
        self.landscape_id = landscape_id
        self.start_id = start_id

    def reset(self, sequences = None) :
        self.model_sequences = {}
        self.measured_sequences = {}
        self.cost = 0
        self.evals = 0
        if sequences:
            self.update_model(sequences)

    def get_min_distance(self,sequence):
        new_dist = 1000
        closest = None
        for seq in self.measured_sequences:
            dist=editdistance.eval(sequence,seq)
            if dist==1:
               new_dist=1
               closest = seq
               break
            else:
                if dist < new_dist:
                   new_dist = dist
                   closest = seq
        #print (new_dist, closest)
        return new_dist, closest


    def add_noise(self,sequence,distance, neighbor_seq):
        signal = self.oracle.get_fitness(sequence)
        #neighbor_seq_fitness = self.oracle.get_fitness(neighbor_seq) 
        # noise = np.random.normal(distance * self._bias * neighbor_seq_fitness,
        #  scale = neighbor_seq_fitness* min(self.max_uncertainty,distance)**2)
        noise = neighbor_seq_fitness*(1+np.random.normal(distance * self._bias , scale = min(self.max_uncertainty, distance)**2))
        #noise = neighbor_seq_fitness*(1+np.random.normal(self._bias , scale = self.max_uncertainty**2))

        alpha = (self.ss) #** distance 
        return signal,noise,alpha


    def _fitness_function(self,sequence):
        if self.ss < 1:
            distance, neighbor_seq = self.get_min_distance(sequence)
            signal,noise,alpha = self.add_noise(sequence,distance,neighbor_seq)
            surrogate_fitness = signal * alpha + noise * (1 - alpha)
            
        else:
            surrogate_fitness = self.oracle.get_fitness(sequence)

        return surrogate_fitness


    def measure_true_landscape(self,sequences):
        for sequence in sequences:
            if sequence not in self.measured_sequences:
                    self.cost += 1
                    self.measured_sequences[sequence]=self.oracle.get_fitness(sequence)

        self.model_sequences = {} #empty cache

    def update_model(self,new_sequences):
        self.measure_true_landscape(new_sequences)

    def get_fitness(self,sequence):

        if sequence in self.measured_sequences: 
            return self.measured_sequences[sequence]
        elif sequence in self.model_sequences and self.cache: #caching model answer to save computation
            return self.model_sequences[sequence]

        else:
            self.model_sequences[sequence] = self._fitness_function(sequence)
            self.evals += 1
            return self.model_sequences[sequence]   



