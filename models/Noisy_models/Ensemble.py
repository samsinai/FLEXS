import numpy as np
from meta.model import Model

class Ensemble_models(Model):
      def __init__(self,list_of_models=None):
         self.cost = 0
         self.evals = 0
         self.measured_sequences = {}
         if list_of_models:
            self.models = list_of_models

      @property
      def model_type(self):
          return ("_").join(model.model_type for model in self.models)

      def add_model(self, model):
          self.models.append(model)

      def reset(self, sequences=None):
         self.cost = 0
         for model in self.models:
             if sequences:
                model.reset(sequences)
             else: 
                model.reset()

      def update_model(self,sequences):
          for model in self.models:
              model.update_model(sequences)

      def get_r2s(self):
          r2s=[]
          for model in self.models:
              r2s.append(model.r2)   
          return r2s  

      def get_fitness(self, sequence):
          fitnesses = self.get_fitness_distribution(sequence)
          self.measured_sequences = self.models[0].measured_sequences
          return np.mean(fitnesses)

      def get_fitness_distribution(self,sequence):
          fitnesses = []
          for model in self.models:
              fitnesses.append(model.get_fitness(sequence))
          self.evals += len(self.models)
          return fitnesses
