%matplotlib inline
import sys
sys.path.append('../')
import RNA 
from utils.sequence_utils import generate_random_mutant
from utils.model_architectures import Linear, NLNN, CNNa
from models.Noisy_models.Neural_network_models import NN_model
from models.Ground_truth_oracles.RNA_landscape_models import RNA_landscape_constructor
from models.Noisy_models.Ensemble import Ensemble_models
from evaluators.Evaluator import Evaluator
from models.Ground_truth_oracles.TF_binding_landscape_models import *
from explorers.bo_explorer import BO_Explorer
from explorers.dqn_explorer import DQN_Explorer 

LANDSCAPE_TYPES_TF = {"RNA": [], 
                      "TF": [
                                "POU3F4_REF_R1",
                                "PAX3_G48R_R1",
                                "SIX6_REF_R1",
                                "VAX2_REF_R1",
                                "VSX1_REF_R1",
                            ]}

import copy
import os
import sys
import random
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from explorers.base_explorer import Base_explorer
from utils.sequence_utils import *

import numpy as np
from bisect import bisect_left

# new BO stuff
from scipy.stats import norm
from scipy.optimize import minimize

class New_BO_Explorer(Base_explorer):
    """
    Bayesian optimization explorer. Uses Gaussian process with Matern kernel
    on black box function.

    Reference: http://krasserm.github.io/2018/03/21/bayesian-optimization/
    """
    def __init__(
        self,
        batch_size=100,
        alphabet="UCGA",
        virtual_screen=10,
        path="./simulations/",
        debug=False,
        method="EI",
    ):
        super(New_BO_Explorer, self).__init__(
            batch_size=batch_size,
            alphabet=alphabet,
            virtual_screen=virtual_screen,
            path=path,
            debug=debug,
        )
        self.explorer_type = "New_BO_Explorer"
        self.alphabet_len = len(alphabet)
        self.method = method
        self.best_fitness = 0
        self.top_sequence = []

        # Gaussian Process with Matern kernel
        matern = Matern(length_scale=1.0, nu=2.5)
        noise = 0.02
        self.gpr = GaussianProcessRegressor(kernel=matern, alpha=noise**2)
        
        # Sampled sequences X and the values y gained from querying the black-box
        self.X_sample = []
        self.y_sample = []
        
    def _one_hot_to_num(self, one_hot):
        return np.argmax(np.transpose(one_hot), axis=1)

    def _initialize(self):
        start_sequence = list(self.model.measured_sequences)[0]
        self.state = translate_string_to_one_hot(start_sequence, self.alphabet)
        self.seq_len = len(start_sequence)
        self._seed_gaussian_process()
        
    def _seed_gaussian_process(self):
        num_samples = 100
        print(f"Seeding GP with {num_samples} samples.")
        random_sequences = generate_random_sequences(
            self.seq_len, num_samples, self.alphabet
        )
        random_states = [translate_string_to_one_hot(seq, self.alphabet) for seq in random_sequences]
        random_states_num = [self._one_hot_to_num(state) for state in random_states]
        fitnesses = [self.model.get_fitness(seq) for seq in random_sequences]
        
        self.X_sample = random_states_num
        self.y_sample = fitnesses

    def reset(self):
        self.X_sample = []
        self.y_sample = []
        
        self.best_fitness = 0
        self.batches = {-1: ""}
        self._reset = True
        
    def EI(self, X, xi=0.01):
        """
        Computes Expected Improvement at X from GP posterior.
        
        Args:
            X: Points to calculate EI at.
            xi: Exploration-exploitation trade-off.
        """
        
        X = self._one_hot_to_num(X)
        mu, sigma = self.gpr.predict(X.reshape(1, -1), return_std=True)
        mu_sample = self.gpr.predict(self.X_sample)

        sigma = sigma.reshape(-1, 1)

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [...]
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei
    
    def propose_new_sequence(self, acq_fn):
        """
        Proposes a new sequence by maximizing the acquisition function.
        
        Args:
            acq_fn: Acquisition function (for example, EI).
        """
        
        print("Enumerating all sequences in the space.")
        
        self.max_val = -float("inf")
        self.max_seq, self.max_state = None, None
        
        def enum_and_eval(curr_seq):
            # if we have a full sequence, then let's evaluate
            if len(curr_seq) == self.seq_len:
                curr_state = translate_string_to_one_hot(curr_seq, self.alphabet)
                v = acq_fn(curr_state)
                if v > self.max_val:
                    self.max_val = v
                    self.max_seq = curr_seq
                    self.max_state = curr_state
            else:
                for char in list(self.alphabet):
                    enum_and_eval(curr_seq + char)
        enum_and_eval("")
                    
        return self.max_seq

    def Thompson_sample(self, measured_batch):
        fitnesses = np.cumsum([np.exp(10 * x[0]) for x in measured_batch])
        fitnesses = fitnesses / fitnesses[-1]
        x = np.random.uniform()
        index = bisect_left(fitnesses, x)
        sequences = [x[1] for x in measured_batch]
        return sequences[index]

    def propose_samples(self):
        if self._reset:
            # indicates model was reset
            self._initialize()
        else:
            # set state to best measured sequence from prior batch
            last_batch = self.batches[self.get_last_batch()]
            measured_batch = sorted(
                [(self.model.get_fitness(seq), seq) for seq in last_batch]
            )
            sampled_seq = self.Thompson_sample(measured_batch)
            self.state = translate_string_to_one_hot(sampled_seq, self.alphabet)
            initial_seq = self.state.copy()
            
        self._reset = False
            
        samples = set()
        
        # fit GPR to samples
        self.gpr.fit(self.X_sample, self.y_sample)

        new_seq = self.propose_new_sequence(self.EI)
        new_state = translate_string_to_one_hot(new_seq, self.alphabet)

        fitness = self.model.get_fitness(new_seq)
        if new_seq not in self.model.measured_sequences:
            if fitness >= self.best_fitness:
                print("New top sequence:", new_seq, fitness)
                self.top_sequence.append((fitness, new_state, self.model.cost))
                self.best_fitness = fitness

        samples.add(new_seq)
        print("Chosen sequence:", new_seq, fitness)
        print("Current best fitness:", self.best_fitness)

        self.X_sample = np.vstack((self.X_sample, self._one_hot_to_num(new_state)))
        self.y_sample.append(fitness)

        return list(samples)
    
bo_explorer_prod = New_BO_Explorer()
bo_explorer_prod.debug = False

evaluator_bo=Evaluator(bo_explorer_prod,landscape_types=LANDSCAPE_TYPES_TF, path="../simulations/eval/")
evaluator_bo.evaluate_for_landscapes(evaluator_bo.consistency_robustness_independence, num_starts=5)