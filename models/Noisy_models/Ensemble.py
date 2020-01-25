import numpy as np
from meta.model import Model

class Ensemble_models(Model):
      def __init__(self,list_of_models=None):
         if list_of_models:
            self.models = list_of_models

      @property
      def model_type(self):
          return "ENS_"+("_").join(model.model_type for model in self.models)

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
      
      def add_model(self, model):
          self.list_of_models.append(model)

      def reset(self, sequences=None):
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
          return np.mean(fitnesses)

      def get_fitness_distribution(self,sequence):
          fitnesses = []
          for model in self.models:
              fitnesses.append(model.get_fitness(sequence))
          return fitnesses
