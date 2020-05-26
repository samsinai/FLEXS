import sys

import numpy as np
import random
import json
from meta.model import Ground_truth_oracle
from utils.multi_dimensional_model import Multi_dimensional_model

AAV2_WT="""MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLD\
KGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQ\
AKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDAD\
SVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVI\
TTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLI\
NNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQG\
CLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPF\
HSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPG\
PCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVL\
IFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGV\
LPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTT\
FSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVY\
SEPRPIGTRYLTRNL"""


class Additive_landscape_constructor():
    def __init__(self):
       self.loaded_landscapes = None

    def load_landscapes(self, config_file, landscapes_to_test = None):
        self.config_file = config_file



        for phenotype in self.loaded_landscapes:
            print (f'{phenotype} landscape loaded')


        
    def construct_landscape_object(self, phenotype):

        landscape = Additive_landscape_AAV(self.config_file, phenotype, **self.loaded_phenotypes)

        return {"landscape_id": phenotype, "starting_seqs": "AAV2" , "landscape_oracle": landscape} 

    def generate_from_loaded_landscapes(self):

        for landscape in self.loaded_phenotypes:

            yield self.construct_landscape_object(landscape)


class Additive_landscape_AAV(Ground_truth_oracle):
    def __init__(self,config_file="../data/AAV_Additive_landscapes/AAV2_single_subs.json",
        phenotype='heart',
        noise=0,
        minimum_fitness_multiplier=1, 
        start=0, 
        end=735):

        self.sequences = {}
        self.phenotype = f'log2_{phenotype}_v_wt'

        self.noise = noise
        self.mfm = minimum_fitness_multiplier
        self.start = start
        self.end = end
        with open(config_file,'r') as infile:
            self.loaded_data = json.load(infile)

        self.data = {}
        for key in self.loaded_data: #convert keys to integers and restrict them to the scope
            new_key = int(key)
            if new_key >= self.start and new_key< self.end: 
               self.data[new_key] = self.loaded_data[key]

        self.top_seq, self.max_possible = self.compute_max_possible()

    def compute_max_possible(self):
        best_seq = ""
        max_fitness = 0 
        for pos in self.data:
            current_max = -10
            current_best = 'M'
            for aa in self.data[pos]:
                current_fit = self.data[pos][aa][self.phenotype]
                if current_fit > current_max and self.data[pos][aa]['log2_packaging_v_wt']>-6:
                   current_best = aa
                   current_max = current_fit
            
            best_seq += current_best
            max_fitness += current_max
        return best_seq, max_fitness

    def _get_raw_fitness(self,seq):
        total_fitness = 0
        for i,s in enumerate(seq):
            if s not in self.data[i+self.start] or self.data[i+self.start][s]['log2_packaging_v_wt']==-6:
               return 0
            else:
               total_fitness += self.data[i+self.start][s][self.phenotype]
            
        return total_fitness + self.mfm * self.max_possible


    def _fitness_function(self,seq):
        normed_fitness = self._get_raw_fitness(seq)/(self.max_possible * (self.mfm+1)) 
        return max(0, normed_fitness)         

    def get_fitness(self,sequence):
         if self.noise == 0:
           if sequence in self.sequences:
              return self.sequences[sequence]
           else:
              self.sequences[sequence] = self._fitness_function(sequence)
              return self.sequences[sequence]
         else:
              self.sequences[sequence] = max(0, min(1, self._fitness_function(sequence)+np.random.normal(scale = self.noise)))
         return self.sequences[sequence]


