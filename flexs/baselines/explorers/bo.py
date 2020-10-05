"""BO explorer."""
from bisect import bisect_left
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import flexs
from flexs.utils.replay_buffers import PrioritizedReplayBuffer
from flexs.utils.sequence_utils import (
    construct_mutant_from_sample,
    generate_random_sequences,
    one_hot_to_string,
    string_to_one_hot,
)


class BO(flexs.Explorer):
    """
    Evolutionary Bayesian Optimization (Evo_BO) explorer.

    Algorithm works as follows:
        for N experiment rounds
            recombine samples from previous batch if it exists and measure them,
                otherwise skip
            Thompson sample starting sequence for new batch
            while less than B samples in batch
                Generate `model_queries_per_batch/sequences_batch_size` samples
                If variance of ensemble models is above twice that of the starting
                    sequence
                Thompson sample another starting sequence
    """

    def __init__(
        self,
        model: flexs.Model,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        log_file: Optional[str] = None,
        method: str = "EI",
        recomb_rate: float = 0,
    ):
        """
        Args:
            method (equal to EI or UCB): The improvement method used in BO,
                default EI.
            recomb_rate: The recombination rate on the previous batch before
                BO proposes samples, default 0.

        """
        name = f"BO_method={method}"
        if not isinstance(model, flexs.Ensemble):
            model = flexs.Ensemble([model], combine_with=lambda x: x)

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
        self.num_actions = 0
        self.state = None
        self.seq_len = None
        self.memory = None
        self.initial_uncertainty = None

    def initialize_data_structures(self):
        """Initialize."""
        self.state = string_to_one_hot(self.starting_sequence, self.alphabet)
        self.seq_len = len(self.starting_sequence)
        # use PER buffer, same as in DQN
        self.memory = PrioritizedReplayBuffer(
            len(self.alphabet) * self.seq_len, 100000, self.sequences_batch_size, 0.6
        )

    def train_models(self):
        """Train the model."""
        if len(self.memory) >= self.sequences_batch_size:
            batch = self.memory.sample_batch()
        else:
            self.memory.batch_size = len(self.memory)
            batch = self.memory.sample_batch()
            self.memory.batch_size = self.sequences_batch_size
        states = batch["next_obs"]
        state_seqs = [
            one_hot_to_string(state.reshape((-1, len(self.alphabet))), self.alphabet)
            for state in states
        ]
        rewards = batch["rews"]
        self.model.train(state_seqs, rewards)

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
        """Compute expected improvement."""
        return np.mean([max(val - self.best_fitness, 0) for val in vals])

    @staticmethod
    def UCB(vals):
        """Upper confidence bound."""
        discount = 0.01
        return np.mean(vals) - discount * np.std(vals)

    def sample_actions(self):
        """Sample actions resulting in sequences to screen."""
        actions = set()
        pos_changes = []
        for pos in range(self.seq_len):
            pos_changes.append([])
            for res in range(len(self.alphabet)):
                if self.state[pos, res] == 0:
                    pos_changes[pos].append((pos, res))

        while len(actions) < self.model_queries_per_batch / self.sequences_batch_size:
            action = []
            for pos in range(self.seq_len):
                if np.random.random() < 1 / self.seq_len:
                    pos_tuple = pos_changes[pos][
                        np.random.randint(len(self.alphabet) - 1)
                    ]
                    action.append(pos_tuple)
            if len(action) > 0 and tuple(action) not in actions:
                actions.add(tuple(action))
        return list(actions)

    def pick_action(self, all_measured_seqs):
        """Pick action."""
        state = self.state.copy()
        actions = self.sample_actions()
        actions_to_screen = []
        states_to_screen = []
        for i in range(self.model_queries_per_batch // self.sequences_batch_size):
            x = np.zeros((self.seq_len, len(self.alphabet)))
            for action in actions[i]:
                x[action] = 1
            actions_to_screen.append(x)
            state_to_screen = construct_mutant_from_sample(x, state)
            states_to_screen.append(one_hot_to_string(state_to_screen, self.alphabet))
        ensemble_preds = self.model.get_fitness(states_to_screen)
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
        if new_state_string not in all_measured_seqs:
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

    def propose_sequences(
        self, measured_sequences: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        if self.num_actions == 0:
            # indicates model was reset
            self.initialize_data_structures()
        else:
            # set state to best measured sequence from prior batch
            last_round_num = measured_sequences["round"].max()
            last_batch = measured_sequences[
                measured_sequences["round"] == last_round_num
            ]
            _last_batch_seqs = last_batch["sequence"].tolist()
            _last_batch_true_scores = last_batch["true_score"].tolist()
            last_batch_seqs = _last_batch_seqs
            if self.recomb_rate > 0 and len(last_batch) > 1:
                last_batch_seqs = self._recombine_population(last_batch_seqs)
            measured_batch = []
            for seq in last_batch_seqs:
                if seq in _last_batch_seqs:
                    measured_batch.append(
                        (_last_batch_true_scores[_last_batch_seqs.index(seq)], seq)
                    )
                else:
                    measured_batch.append((np.mean(self.model.get_fitness([seq])), seq))
            measured_batch = sorted(measured_batch)
            sampled_seq = self.Thompson_sample(measured_batch)
            self.state = string_to_one_hot(sampled_seq, self.alphabet)
        # generate next batch by picking actions
        self.initial_uncertainty = None
        samples = set()
        prev_cost = self.model.cost
        all_measured_seqs = set(measured_sequences["sequence"].tolist())
        while self.model.cost - prev_cost < self.model_queries_per_batch:
            uncertainty, new_state_string, _ = self.pick_action(all_measured_seqs)
            all_measured_seqs.add(new_state_string)
            samples.add(new_state_string)
            if self.initial_uncertainty is None:
                self.initial_uncertainty = uncertainty
            if uncertainty > 2 * self.initial_uncertainty:
                # reset sequence to starting sequence if we're in territory that's too
                # uncharted
                sampled_seq = self.Thompson_sample(measured_batch)
                self.state = string_to_one_hot(sampled_seq, self.alphabet)
                self.initial_uncertainty = None

        if len(samples) < self.sequences_batch_size:
            random_sequences = generate_random_sequences(
                self.seq_len, self.sequences_batch_size - len(samples), self.alphabet
            )
            samples.update(random_sequences)
        # get predicted fitnesses of samples
        samples = list(samples)
        preds = np.mean(self.model.get_fitness(samples), axis=1)
        # train ensemble model before returning samples
        self.train_models()

        return samples, preds


class GPR_BO(flexs.Explorer):
    """Explorer using GP-based Bayesian Optimization.

    Uses Gaussian process with RBF kernel on black box function.
    IMPORTANT: This explorer is not limited by any virtual screening restriction,
    and is used to find the unrestricted performance of Bayesian Optimization
    techniques in small landscapes.

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
        seq_proposal_method="Thompson",
    ):
        """Initialize the explorer."""
        name = f"GPR_BO_Explorer-seq_proposal_method={seq_proposal_method}"
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
        self.seq_proposal_method = seq_proposal_method
        self.best_fitness = 0
        self.top_sequence = []

        self.seq_len = len(starting_sequence)
        self.maxima = None

    def reset(self):
        """Reset."""
        self.best_fitness = 0
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
                mus = self.model.get_fitness(curr_seq)
                mu, sigma = np.mean(mus), np.std(mus)
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
                mus = self.model.get_fitness(curr_seq)
                mu = np.mean(mus)
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
                mus = self.model.get_fitness(curr_seq)
                mu, sigma = np.mean(mus), np.std(mus)
                self.maxima.append([mu + 0.01 * sigma, curr_seq])
            else:
                for char in list(self.alphabet):
                    enum_and_eval(curr_seq + char)

        enum_and_eval("")

        # Sort descending based on the value.
        return sorted(self.maxima, reverse=True, key=lambda x: x[0])

    def propose_sequences(
        self, measured_sequences: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose `batch_size` samples."""
        samples = set()
        # TODO: Add UCB and Thompson proposal methods
        seq_proposal_funcs = {
            "Greedy": self.propose_sequences_via_greedy,
            "Thompson": self.propose_sequences_via_thompson,
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
                new_state = string_to_one_hot(new_seq, self.alphabet)
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
