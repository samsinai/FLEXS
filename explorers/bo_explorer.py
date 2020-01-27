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
                 method = 'EI'):
        super(BO_Explorer, self).__init__(batch_size=batch_size, alphabet=alphabet, virtual_screen=virtual_screen, path=path, debug=debug)
        self.explorer_type='BO_Ensemble'
        self.alphabet_len = len(alphabet)
        self.method = method
        self.batch_size = batch_size 
        self.virtual_screen = virtual_screen
        self.best_fitness = 0
        self.top_sequence = []
        self.num_actions = 0
        # use PER buffer, same as in DQN 
        self.model_type = 'blank'

    def initialize_data_structures(self):
        start_sequence = list(self.model.measured_sequences)[0]
        self.state = translate_string_to_one_hot(start_sequence, self.alphabet)
        self.seq_len = len(start_sequence)
        self.memory = PrioritizedReplayBuffer(self.alphabet_len*self.seq_len, 
                                              100000, self.batch_size, 0.6) 

    def reset(self):
        self.num_actions = 0
    
    def train_models(self):
        batch = self.memory.sample_batch()
        states = batch['next_obs']
        state_seqs = [translate_one_hot_to_string(state.reshape((-1, self.seq_len)), self.alphabet) 
            for state in states]
        self.model.update_model(state_seqs)
                
    def EI(self, vals):
        return np.mean([max(val - self.best_fitness, 0) for val in vals])
        
    def UCB(self, vals):
        discount = 0.01 
        return np.mean(vals) - discount*np.std(vals)
                   
    def pick_action(self):
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
        self.num_actions += 1
        return new_state_string, reward 

    def propose_samples(self):
        if self.num_actions == 0:
            # indicates model was reset 
            self.initialize_data_structures()
        samples = []
        for _ in range(self.batch_size * self.virtual_screen):
            new_state_string, reward = self.pick_action()
            samples.append((reward, new_state_string))
        samples = sorted(set(samples))[-self.batch_size:]
        samples = [sample[1] for sample in samples]
        for _ in range(self.batch_size - len(samples)):
            # if we still do not have enough 
            new_state_string, reward = self.pick_action()
            samples.append(new_state_string) 
        return samples 