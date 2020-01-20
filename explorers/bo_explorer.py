import copy 
import os
import sys
import random 
from collections import defaultdict 
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, BatchNormalization, Flatten, concatenate, Lambda
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.layers import (
    Input,
    Embedding,
    Dropout,
    Conv1D,
    Lambda,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Concatenate,
    SpatialDropout1D,
    Dense,
    Activation,
    BatchNormalization,
    Concatenate,
    Reshape,
    Multiply,
    Add,
    Dot,
    Flatten,
    Lambda,
    LSTM,
    GRU,
    Bidirectional
)
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l1, l2

from explorers.base_explorer import Base_explorer 
from utils.sequence_utils import *
from utils.replay_buffers import PrioritizedReplayBuffer 

class BO_Explorer(Base_explorer):
    def __init__(self, batch_size=100, alphabet='UCGA',
                 virtual_screen=10, path="./simulations/", debug=False, 
                 seq_len=40, method = 'EI'):
        super(BO_Explorer, self).__init__(batch_size=batch_size, alphabet=alphabet, virtual_screen=virtual_screen, path=path, debug=debug)
        self.explorer_type='BO_Ensemble'
        self.alphabet_len = len(alphabet)
        self.seq_len = seq_len
        self.method = method
        self.batch_size = batch_size 
        self.virtual_screen = virtual_screen
        self.best_fitness = 0
        self.top_sequence = []
        # use PER buffer, same as in DQN 
        self.model_type = 'blank'
    
    def train_models(self):
        for i in range(len(self.model.models)):
            batch = self.memory.sample_batch()
            states = batch['next_obs']
            self.model.models[i].update_model(states)
                
    def EI(self, vals):
        return np.mean([max(val - self.best_fitness, 0) for val in vals])
        
    def UCB(self, vals):
        discount = 0.01 
        return np.mean(vals) - discount*np.std(vals)

    def initialize_data_structures(self):
        start_sequence = list(self.model.measured_sequences)[0]
        self.state = translate_string_to_one_hot(start_sequence, self.alphabet)
        self.seq_len = len(start_sequence)
        self.memory = PrioritizedReplayBuffer(self.alphabet_len*self.seq_len, 
                                              100000, self.batch_size, 0.6) 
                   
    def pick_action(self):
        if self.model.cost == 0:
            # if model/start sequence got reset
            self.initialize_data_structures()
        state = self.state.copy()
        possible_actions = [(i, j) for i in range(self.alphabet_len) 
                            for j in range(self.seq_len) if state[i, j] == 0]
        action_inds = np.random.choice(len(possible_actions), 
                                   size=self.virtual_screen, 
                                   replace=False)
        actions = [possible_actions[ind] for ind in action_inds]
        actions_to_screen = []
        states_to_screen = []
        for i in range(self.virtual_screen):
            x = np.zeros((self.alphabet_len, self.seq_len))
            x[actions[i]] = 1 
            actions_to_screen.append(x)
            state_to_screen = construct_mutant_from_sample(x, state)
            states_to_screen.append(translate_one_hot_to_string(state_to_screen, self.alphabet))
        ensemble_preds = [self.model.get_fitness_distribution(state) for state in states_to_screen]
        method_pred = [self.EI(vals) for vals in ensemble_preds] if self.method == 'EI' \
            else [self.UCB(vals) for vals in ensemble_preds]
        action_ind = np.argmax(method_pred)
        action = actions_to_screen[action_ind]
        new_state_string = states_to_screen[action_ind]
        self.state = translate_string_to_one_hot(new_state_string, self.alphabet)
        new_state = self.state
        reward = np.mean(ensemble_preds[action_ind])
        if not new_state_string in self.model.measured_sequences:
            if reward >= self.best_fitness:
                self.top_sequence.append((reward, new_state, self.model.cost))
            self.best_fitness = max(self.best_fitness, reward)
            self.memory.store(state.ravel(), action.ravel(), reward, new_state.ravel())
        if self.model.cost % self.batch_size == 0 and self.model.cost > 0:
            self.train_models()

    '''       
    def propose_samples(self):
        samples = []
        init_cost = copy.deepcopy(self.model.cost) 
        total_proposed = 0
        # try to make as many new sequences as possible, then fill the remainder up 
        while (self.model.cost - init_cost) < self.batch_size and \
            total_proposed < self.batch_size*self.virtual_screen - (self.model.cost - init_cost):
            cost = copy.deepcopy(self.model.cost) 
            self.pick_action()
            if cost != self.model.cost: # new sequence
                samples.append(translate_one_hot_to_string(self.state, self.alphabet))
            total_proposed += 1
        new_seq_made = (self.model.cost - init_cost)
        for _ in range(self.batch_size - new_seq_made):
            self.pick_action()
            samples.append(translate_one_hot_to_string(self.state, self.alphabet))
        return samples 
    '''
    def propose_samples(self):
        samples = []
        if self.model.model_type != self.model_type:
            # indicates model has been reset 
            self.model_type = self.model.model_type
            self.initialize_data_structures()
        for _ in range(self.batch_size):
            self.pick_action()
            samples.append(translate_one_hot_to_string(self.state,self.alphabet))
        return samples 