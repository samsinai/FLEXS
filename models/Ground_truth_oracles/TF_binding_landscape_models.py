from meta.model import Ground_truth_oracle
import pandas as pd
from glob import glob 


class TF_binding_landscape_constructor():
    def __init__(self):
       self.loaded_landscapes=[]
       self.starting_seqs = {}

    def load_landscapes(self,data_path="data/TF_binding_landscapes", landscapes_to_test = None):
        start_seqs= pd.read_csv(f'{data_path}/TF_starting_id.csv', sep=",")
        for i, row in start_seqs.iterrows():
            self.starting_seqs[row["id"]] = row["start"] 

        self.loaded_landscapes =   glob(f'{data_path}/landscapes/*')

        if landscapes_to_test!="all":
          if landscapes_to_test: 
              self.loaded_landscapes = [self.loaded_landscapes[i] for i in landscapes_to_test]
          else:
              self.loaded_landscapes=[]
            
        print(f'{len(self.loaded_landscapes)} TF landscapes loaded.')
        
    def construct_landscape_object(self,landscape_file):

        landscape_id = landscape_file.strip("data/TF_binding_landscapes/landscapes/").strip("_8mers.txt")
        landscape = TF_binding_landscape()
        landscape.construct(landscape_file)

        return {"landscape_id": landscape_id, "starting_seqs": self.starting_seqs, "landscape_oracle": landscape} 

    def generate_from_loaded_landscapes(self):

        for landscape_file in self.loaded_landscapes:

            yield self.construct_landscape_object(landscape_file)


class TF_binding_landscape(Ground_truth_oracle):

    def __init__(self):
       self.sequences = {}
       
    def construct(self,landscape_file):
        data = pd.read_csv(f'{landscape_file}',sep="\t")
        for i,row in data.iterrows():
            self.sequences[row["8-mer"]] = row["E-score"]
            self.sequences[row["8-mer.1"]] = row["E-score"]

    def get_fitness(self,sequence):
        return self.sequences[sequence]

