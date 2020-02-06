import copy 
import os
import sys
import random 
from collections import defaultdict 
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from explorers.base_explorer import Base_explorer 
from utils.sequence_utils import *
from utils.replay_buffers import PrioritizedReplayBuffer 

class BO_Explorer(Base_explorer):
    def __init__(self, batch_size=100, alphabet='UCGA',
                 virtual_screen=10, path="./simulations/", debug=False, 
                 method = 'EI'):
        super(BO_Explorer, self).__init__(batch_size=batch_size, alphabet=alphabet, virtual_screen=virtual_screen, path=path, debug=debug)
        self.explorer_type='BO_Explorer'
        self.alphabet_len = len(alphabet)
        self.method = method
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
        self.best_fitness = 0
        self.batches = {-1:""}
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

    def sample_actions(self):
        actions, actions_set = [], set()
        pos_changes = []
        for i in range(self.seq_len):
            pos_changes.append([])
            for j in range(self.alphabet_len):
                if self.state[j, i] == 0:
                    pos_changes[i].append((j, i))

        while len(actions_set) < self.virtual_screen:
            action = []
            for i in range(self.seq_len):
                if np.random.random() < 1/self.seq_len:
                    pos_tuple = pos_changes[i][np.random.randint(self.alphabet_len - 1)]
                    action.append(pos_tuple)
            if len(action) > 0 and tuple(action) not in actions_set:
                actions_set.add(tuple(action))
                actions.append(tuple(action))
        return actions 
                   
    def pick_action(self):
        state = self.state.copy()
        actions = self.sample_actions()
        actions_to_screen = []
        states_to_screen = []
        for i in range(self.virtual_screen):
            x = np.zeros((self.alphabet_len, self.seq_len))
            for action in actions[i]:
                x[action] = 1
            actions_to_screen.append(x)
            state_to_screen = construct_mutant_from_sample(x, state)
            states_to_screen.append(translate_one_hot_to_string(state_to_screen, self.alphabet))
        ensemble_preds = [self.model.get_fitness_distribution(state) for state in states_to_screen]
        method_pred = [self.EI(vals) for vals in ensemble_preds] if self.method == 'EI' \
            else [self.UCB(vals) for vals in ensemble_preds]
        action_ind = np.argmax(method_pred)
        uncertainty = np.std(method_pred[action_ind])
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
        self.num_actions += 1
        return uncertainty, new_state_string, reward 

    def propose_samples(self):
        if self.num_actions == 0:
            # indicates model was reset 
            self.initialize_data_structures()
        else:
            # set state to best measured sequence from prior batch 
            last_batch = self.batches[self.get_last_batch()]
            measured_batch = sorted([(self.model.get_fitness(seq), seq) for seq in last_batch])
            _, best_seq = measured_batch[-1]
            self.state = translate_string_to_one_hot(best_seq, self.alphabet)
            initial_seq = self.state.copy()
        # generate next batch by picking actions 
        self.initial_uncertainty = None 
        samples = set()
        prev_cost, prev_evals = copy.deepcopy(self.model.cost), copy.deepcopy(self.model.evals)
        while (self.model.cost - prev_cost < self.batch_size) and (self.model.evals - prev_evals < self.batch_size * self.virtual_screen):
            uncertainty, new_state_string, reward = self.pick_action()
            samples.add(new_state_string)  
            if self.initial_uncertainty is None:
                self.initial_uncertainty = uncertainty 
            if uncertainty > 2 * self.initial_uncertainty:
                # reset sequence to starting sequence if we're in territory that's too uncharted
                self.state = initial_seq.copy()
        if len(samples) < self.batch_size:
            random_sequences = generate_random_sequences(self.seq_len, self.batch_size - len(samples), self.alphabet)   
            samples.update(random_sequences)      
        # train ensemble model before returning samples 
        self.train_models()

        return list(samples) 