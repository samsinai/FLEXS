import numpy as np
import random
from meta.model import Model


class Ensemble_models(Model):
    def __init__(self, list_of_models=None, adaptive=True):
        if list_of_models:
            self.models = list_of_models
            self.weighted_r2s = [
                1.0 / len(list_of_models) for i in range(len(list_of_models))
            ]

        self.adaptive = adaptive
        if self.adaptive:
            self.name = "AdaENS"
        else:
            self.name = "ENS"

    @property
    def model_type(self):
        return f"{self.name}_{len(self.models)}_" + ("_").join(
            set([model.model_type for model in self.models])
        )

    @property
    def measured_sequences(self):
        measured_sequences_out = {}
        for model in self.models:
            measured_sequences_out.update(model.measured_sequences)
        return measured_sequences_out

    @property
    def cost(self):
        return len(self.measured_sequences)

    @property
    def evals(self):
        return np.mean([model.evals for model in self.models])

    @property
    def landscape_id(self):
        return self.models[0].landscape_id

    @property
    def start_id(self):
        return self.models[0].start_id

    @property
    def num_models(self):
        return len(self.models)

    def add_model(self, model):
        self.list_of_models.append(model)
        self.r2s = [
            1 / len(self.list_of_models) for i in range(len(self.list_of_models))
        ]

    def reset(self, sequences=None):
        for model in self.models:
            if sequences:
                model.reset(sequences)
            else:
                model.reset()

    def update_model(self, sequences):  # , bootstrap= True):
        for model in self.models:
            # if bootstrap and len(sequences)>20:
            #    sub_sequences = random.sample(sequences, int(len(sequences)*1))
            #    model.update_model(sub_sequences)
            # else:
            model.update_model(sequences)
        try:
            self.r2s = self.get_r2s()
            self.weighted_r2s = self.get_weighted_r2s()
            # print (self.r2s)
            print(self.weighted_r2s)
        except Exception as e:
            print(e)
            print("R^2 not computed for this type of model")

    def get_weighted_r2s(self):
        r2s = []
        for model in self.models:
            r2s.append(max(0.001, model.r2))
        weighted_r2s = [r / sum(r2s) for r in r2s]
        return weighted_r2s

    def get_r2s(self):
        r2s = []
        for model in self.models:
            r2s.append(model.r2)
        self.r2s = r2s
        return r2s

    def get_fitness(self, sequence):
        if sequence in self.measured_sequences:
            return self.measured_sequences[sequence]

        if not self.adaptive:
            fitnesses = self.get_fitness_distribution(sequence)
            return np.mean(fitnesses)
        else:
            fitnesses = self.get_fitness_distribution(sequence)
            weighted_fitnesses = [f * w for f, w in zip(fitnesses, self.weighted_r2s)]
            return np.sum(weighted_fitnesses)

    def get_fitness_distribution(self, sequence):
        fitnesses = []
        for model in self.models:
            fitnesses.append(model.get_fitness(sequence))
        return fitnesses
