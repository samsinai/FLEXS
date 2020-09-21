"""BO explorer."""
import copy
from bisect import bisect_left

import numpy as np
import flexs
import flexs
from flexs.utils.replay_buffers import PrioritizedReplayBuffer
from flexs.utils.sequence_utils import (
    construct_mutant_from_sample,
    generate_random_sequences,
    string_to_one_hot,
    one_hot_to_string,
)


class BO(flexs.Explorer):
    """Explorer using Bayesian Optimization."""

    def __init__(
        self,
        model,
        rounds,
        sequences_batch_size,
        model_queries_per_batch,
        starting_sequence,
        alphabet,
        log_file=None,
        virtual_screen=10,
        method="EI",
        recomb_rate=0,
    ):
        """Bayesian Optimization (BO) explorer.

        Parameters:
            method (str, equal to EI or UCB): The improvement method used in BO,
                default EI.
            recomb_rate (float): The recombination rate on the previous batch before
                BO proposes samples, default 0.

        Algorithm works as follows:
            for N experiment rounds
                recombine samples from previous batch if it exists and measure them,
                    otherwise skip
                Thompson sample starting sequence for new batch
                while less than B samples in batch
                    Generate VS virtual screened samples
                    If variance of ensemble models is above twice that of the starting
                        sequence
                    Thompson sample another starting sequence
        """
        name = "BO-optimization_method={optimization_method}"
        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )
        self.alphabet = alphabet
        self.method = method
        self.recomb_rate = recomb_rate
        self.best_fitness = 0
        self.top_sequence = []
        self.num_actions = 0
        self.virtual_screen = virtual_screen
        # use PER buffer, same as in DQN
        self.model_type = "blank"
        self.state = None
        self.seq_len = None
        self.memory = None
        self.initial_uncertainty = None

    def initialize_data_structures(self):
        """Initialize."""
        self.state = string_to_one_hot(self.starting_sequence, self.alphabet)
        self.seq_len = len(self.starting_sequence)
        self.memory = PrioritizedReplayBuffer(
            len(self.alphabet) * self.seq_len, 100000, self.sequences_batch_size, 0.6
        )

    def reset(self):
        """Reset the explorer."""
        self.best_fitness = 0
        self.batches = {-1: ""}
        self.num_actions = 0

    def train_models(self):
        """Train the model."""
        batch = self.memory.sample_batch()
        states = batch["next_obs"]
        state_seqs = [
            one_hot_to_string(state.reshape((-1, self.seq_len)), self.alphabet)
            for state in states
        ]
        self.model.update_model(state_seqs)

    def _recombine_population(self, gen):
        np.random.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA = []
            strB = []
            switch = False
            for ind in range(len(gen[i])):
                if np.random.random() < self.recomb_rate:
                    switch = not switch

                # putting together recombinants
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])

            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret

    def EI(self, vals):
        """Expected improvement."""
        return np.mean([max(val - self.best_fitness, 0) for val in vals])

    @staticmethod
    def UCB(vals):
        """Upper confidence bound."""
        discount = 0.01
        return np.mean(vals) - discount * np.std(vals)

    def sample_actions(self):
        """Sample actions resulting in sequences to screen."""
        actions, actions_set = [], set()
        pos_changes = []
        for i in range(self.seq_len):
            pos_changes.append([])
            for j in range(len(self.alphabet)):
                if self.state[j, i] == 0:
                    pos_changes[i].append((j, i))

        while len(actions_set) < self.virtual_screen:
            action = []
            for i in range(self.seq_len):
                if np.random.random() < 1 / self.seq_len:
                    pos_tuple = pos_changes[i][
                        np.random.randint(len(self.alphabet) - 1)
                    ]
                    action.append(pos_tuple)
            if len(action) > 0 and tuple(action) not in actions_set:
                actions_set.add(tuple(action))
                actions.append(tuple(action))
        return actions

    def pick_action(self, all_measured_seqs):
        """Pick action."""
        state = self.state.copy()
        actions = self.sample_actions()
        actions_to_screen = []
        states_to_screen = []
        for i in range(self.virtual_screen):
            x = np.zeros((len(self.alphabet), self.seq_len))
            for action in actions[i]:
                x[action] = 1
            actions_to_screen.append(x)
            state_to_screen = construct_mutant_from_sample(x, state)
            states_to_screen.append(one_hot_to_string(state_to_screen, self.alphabet))
        ensemble_preds = [
            self.model.get_fitness_distribution(state) for state in states_to_screen
        ]
        method_pred = (
            [self.EI(vals) for vals in ensemble_preds]
            if self.method == "EI"
            else [self.UCB(vals) for vals in ensemble_preds]
        )
        action_ind = np.argmax(method_pred)
        uncertainty = np.std(method_pred[action_ind])
        action = actions_to_screen[action_ind]
        new_state_string = states_to_screen[action_ind]
        self.state = string_to_one_hot(new_state_string, self.alphabet)
        new_state = self.state
        reward = np.mean(ensemble_preds[action_ind])
        if not new_state_string in all_measured_seqs:
            if reward >= self.best_fitness:
                self.top_sequence.append((reward, new_state, self.model.cost))
            self.best_fitness = max(self.best_fitness, reward)
            self.memory.store(state.ravel(), action.ravel(), reward, new_state.ravel())
        self.num_actions += 1
        return uncertainty, new_state_string, reward

    @staticmethod
    def Thompson_sample(measured_batch):
        """Pick a sequence via Thompson sampling."""
        fitnesses = np.cumsum([np.exp(10 * x[0]) for x in measured_batch])
        fitnesses = fitnesses / fitnesses[-1]
        x = np.random.uniform()
        index = bisect_left(fitnesses, x)
        sequences = [x[1] for x in measured_batch]
        return sequences[index]

    def propose_sequences(self, measured_sequences):
        """Propose `batch_size` samples."""
        if self.num_actions == 0:
            # indicates model was reset
            self.initialize_data_structures()
        else:
            # set state to best measured sequence from prior batch
            last_round_num = measured_sequences["round"].max()
            last_batch = measured_sequences[
                measured_sequences["round"] == last_round_num
            ]
            last_batch_seqs = last_batch["sequence"].values
            if self.recomb_rate > 0 and len(last_batch) > 1:
                last_batch_seqs = self._recombine_population(list(last_batch_seqs))
            measured_batch = sorted(
                [(self.model.get_fitness(seq), seq) for seq in last_batch]
            )
            sampled_seq = self.Thompson_sample(measured_batch)
            self.state = string_to_one_hot(sampled_seq, self.alphabet)
        # generate next batch by picking actions
        self.initial_uncertainty = None
        samples = set()
        prev_cost, prev_evals = (
            copy.deepcopy(self.model.cost),
            copy.deepcopy(self.model.evals),
        )
        all_measured_seqs = set(measured_sequences["sequence"].values)
        while (self.model.cost - prev_cost < self.sequences_batch_size) and (
            self.model.evals - prev_evals
            < self.sequences_batch_size * self.virtual_screen
        ):
            uncertainty, new_state_string, _ = self.pick_action(all_measured_seqs)
            all_measured_seqs.add(new_state_string)
            samples.add(new_state_string)
            if self.initial_uncertainty is None:
                self.initial_uncertainty = uncertainty
            if uncertainty > 2 * self.initial_uncertainty:
                # reset sequence to starting sequence if we're in territory that's too uncharted
                sampled_seq = self.Thompson_sample(measured_batch)
                self.state = string_to_one_hot(sampled_seq, self.alphabet)
                self.initial_uncertainty = None

        if len(samples) < self.sequences_batch_size:
            random_sequences = generate_random_sequences(
                self.seq_len, self.sequences_batch_size - len(samples), self.alphabet
            )
            samples.update(random_sequences)
        # get predicted fitnesses of samples
        preds = [self.model.get_fitness(sample) for sample in samples]
        # train ensemble model before returning samples
        self.train_models()

        return list(samples), preds


class GPR_BO(flexs.Explorer):
    """Explorer using Bayesian Optimization.

    Uses Gaussian process with RBF kernel on black box function.
    IMPORTANT: This explorer is not limited by `virtual_screen`, and is used to find
    the upper-bound performance of Bayesian Optimization techniques.

    Reference: http://krasserm.github.io/2018/03/21/bayesian-optimization/
    """

    def __init__(
        self,
        model,
        rounds,
        sequences_batch_size,
        model_queries_per_batch,
        starting_sequence,
        alphabet,
        log_file=None,
        virtual_screen=10,
        method="EI",
        seq_proposal_method="Thompson",
    ):
        """Initialize the explorer."""
        name = (
            "GPR_BO_Explorer-method={method}-seq_proposal_method={seq_proposal_method}"
        )
        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )
        self.alphabet = alphabet
        self.alphabet_len = len(alphabet)
        self.method = method
        self.seq_proposal_method = seq_proposal_method
        self.best_fitness = 0
        self.top_sequence = []
        self.virtual_screen = virtual_screen

        self.seq_len = None
        self.maxima = None

    def reset(self):
        """Reset."""
        self.best_fitness = 0
        self.batches = {-1: ""}
        self._reset = True

    def propose_sequences_via_thompson(self):
        """Propose a batch of new sequences.

        Based on Thompson sampling with a Gaussian posterior.
        """
        print("Enumerating all sequences in the space.")

        self.maxima = []

        def enum_and_eval(curr_seq):
            # if we have a full sequence, then let's evaluate
            if len(curr_seq) == self.seq_len:
                mu, sigma = self.model.get_fitness(curr_seq, return_std=True)
                estimated_fitness = np.random.normal(mu, sigma)
                self.maxima.append([estimated_fitness, curr_seq])
            else:
                for char in list(self.alphabet):
                    enum_and_eval(curr_seq + char)

        enum_and_eval("")

        # Sort descending based on the value.
        return sorted(self.maxima, reverse=True, key=lambda x: x[0])

    def propose_sequences_via_greedy(self):
        """Propose a batch of new sequences.

        Based on greedy in the expectation of the Gaussian posterior.
        """
        print("Enumerating all sequences in the space.")

        self.maxima = []

        def enum_and_eval(curr_seq):
            # if we have a full sequence, then let's evaluate
            if len(curr_seq) == self.seq_len:
                mu = self.model.get_fitness(curr_seq, return_std=False)
                self.maxima.append([mu, curr_seq])
            else:
                for char in list(self.alphabet):
                    enum_and_eval(curr_seq + char)

        enum_and_eval("")

        # Sort descending based on the value.
        return sorted(self.maxima, reverse=True, key=lambda x: x[0])

    def propose_sequences_via_ucb(self):
        """Propose a batch of new sequences.

        Based on upper confidence bound.
        """
        print("Enumerating all sequences in the space.")

        self.maxima = []

        def enum_and_eval(curr_seq):
            # if we have a full sequence, then let's evaluate
            if len(curr_seq) == self.seq_len:
                mu, sigma = self.model.get_fitness(curr_seq, return_std=True)
                self.maxima.append([mu + 0.01 * sigma, curr_seq])
            else:
                for char in list(self.alphabet):
                    enum_and_eval(curr_seq + char)

        enum_and_eval("")

        # Sort descending based on the value.
        return sorted(self.maxima, reverse=True, key=lambda x: x[0])

    def propose_sequences(self, measured_sequences):
        """Propose `batch_size` samples."""
        samples = set()

        seq_proposal_funcs = {
            "Thompson": self.propose_sequences_via_thompson,
            "Greedy": self.propose_sequences_via_greedy,
            "UCB": self.propose_sequences_via_ucb,
        }
        seq_proposal_func = seq_proposal_funcs[self.seq_proposal_method]
        new_seqs = seq_proposal_func()
        new_states = []
        new_fitnesses = []
        i = 0
        all_measured_seqs = set(measured_sequences["sequence"].values)
        while (len(new_states) < self.sequences_batch_size) and i < len(new_seqs):
            new_fitness, new_seq = new_seqs[i]
            if new_seq not in all_measured_seqs:
                new_state = one_hot_to_string(new_seq, self.alphabet)
                if new_fitness >= self.best_fitness:
                    self.top_sequence.append((new_fitness, new_state, self.model.cost))
                    self.best_fitness = new_fitness
                samples.add(new_seq)
                all_measured_seqs.add(new_seq)
                new_states.append(new_state)
                new_fitnesses.append(new_fitness)
            i += 1

        print("Current best fitness:", self.best_fitness)

        return list(samples), new_fitnesses
