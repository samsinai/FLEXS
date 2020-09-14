"""DQN explorer."""
import copy
import random
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch import optim as optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import flexs
from flexs.utils.replay_buffers import PrioritizedReplayBuffer
from flexs.utils.sequence_utils import (
    construct_mutant_from_sample,
    generate_random_sequences,
    make_random_action,
    renormalize_moves,
    sample_greedy,
    sample_random,
    one_hot_to_string,
    translate_string_to_one_hot,
)


class Q_Network(nn.Module):
    """Q Network implementation, used in DQN Explorer."""

    def __init__(self, sequence_len, alphabet_len):
        """Initialize the Q Network."""
        super(Q_Network, self).__init__()
        self.sequence_len = sequence_len
        self.alphabet_len = alphabet_len
        self.linear1 = nn.Linear(
            2 * alphabet_len * sequence_len, alphabet_len * sequence_len
        )
        self.bn1 = nn.BatchNorm1d(alphabet_len * sequence_len)
        self.linear2 = nn.Linear(alphabet_len * sequence_len, sequence_len)
        self.bn2 = nn.BatchNorm1d(sequence_len)
        self.linear3 = nn.Linear(sequence_len, 1)

    def forward(self, x):  # pylint: disable=W0221
        """Take a forward step."""
        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        x = F.relu(self.linear3(x))
        return x


def build_q_network(sequence_len, alphabet_len, device):
    """Build the Q Network."""
    model = Q_Network(sequence_len, alphabet_len).to(device)
    return model


class DQN_Explorer(flexs.Explorer):
    """Explorer for DQN."""

    def __init__(
        self,
        model,
        landscape,
        rounds,
        starting_sequence,
        sequences_batch_size,
        model_queries_per_batch,
        alphabet,
        batch_size=100,
        virtual_screen=10,
        memory_size=100000,
        train_epochs=20,
        gamma=0.9,
        device="cpu",
        noise_alpha=1,  # pylint: disable=W0613
    ):
        """DQN explorer class.

        DQN Explorer implementation, based off
        https://colab.research.google.com/drive/1NsbSPn6jOcaJB_mp9TmkgQX7UrRIrTi0.

        Algorithm works as follows:
        for N experiment rounds
            collect samples with policy
            policy updates using Q network:
                Q(s, a) <- Q(s, a) + alpha * (R(s, a) + gamma * max Q(s, a) - Q(s, a))

        Attributes:
        memory_size: size of agent memory
        batch_size: batch size to train the PER buffer with
        experiment_batch_size: the batch size of the experiment.
            that is, if this were a lab, this would be the number of sequences
            evaluated in a lab trial
        """
        name = "DQN_Explorer"
        super().__init__(
            model,
            landscape,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
        )
        self.explorer_type = "DQN_Explorer"
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.memory_size = memory_size
        self.gamma = gamma
        self.best_fitness = 0
        self.train_epochs = train_epochs
        self.epsilon_min = 0.1
        self.device = device
        self.top_sequence = []
        self.times_seen = Counter()
        self.num_actions = 0
        self.model_type = "blank"

        self.state = None
        self.seq_len = None
        self.q_network = None
        self.memory = None

    def initialize_data_structures(self):
        """Initialize internal data structures."""
        start_sequence = list(self.model.measured_sequences)[0]
        self.state = translate_string_to_one_hot(start_sequence, self.alphabet)
        self.seq_len = len(start_sequence)
        self.q_network = build_q_network(self.seq_len, len(self.alphabet), self.device)
        self.q_network.eval()
        self.memory = PrioritizedReplayBuffer(
            len(self.alphabet) * self.seq_len, self.memory_size, self.batch_size, 0.6
        )

    def reset(self):
        """Reset the explorer."""
        self.best_fitness = 0
        self.batches = {-1: ""}
        self.num_actions = 0

    def sample(self):
        """Sample a random `batch_size` subset of the memory."""
        indices = np.random.choice(len(self.memory), self.batch_size)
        rewards, actions, states, next_states = zip(
            *[self.memory[ind] for ind in indices]
        )
        return (
            np.array(rewards),
            np.array(actions),
            np.array(states),
            np.array(next_states),
        )

    def calculate_next_q_values(self, state_v):
        """Calculate the next Q values."""
        dim = self.alphabet_size * self.seq_len
        states_repeated = state_v.repeat(1, dim).reshape(-1, dim)
        actions_repeated = torch.FloatTensor(np.identity(dim)).repeat(len(state_v), 1)
        next_states_actions = torch.cat((states_repeated, actions_repeated), 1)
        next_states_values = self.q_network(next_states_actions)
        next_states_values = next_states_values.reshape(len(state_v), -1)

        return next_states_values

    def q_network_loss(self, batch):
        """Calculate MSE.

        Computes between actual state action values, and expected state action values
        from DQN.
        """
        rewards, actions, states, next_states = (
            batch["rews"],
            batch["acts"],
            batch["obs"],
            batch["next_obs"],
        )

        state_action_v = torch.FloatTensor(np.hstack((states, actions)))
        rewards_v = torch.FloatTensor(rewards)
        next_states_v = torch.FloatTensor(next_states)

        state_action_values = self.q_network(state_action_v).view(-1)
        next_state_values = self.calculate_next_q_values(next_states_v)
        next_state_values = next_state_values.max(1)[0].detach()
        expected_state_action_values = next_state_values * self.gamma + rewards_v

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def train_actor(self, train_epochs):
        """Train the Q Network."""
        total_loss = 0.0
        # train Q network on new samples
        optimizer = optim.Adam(self.q_network.parameters())
        for _ in range(train_epochs):
            batch = self.memory.sample_batch()
            optimizer.zero_grad()
            loss = self.q_network_loss(batch)
            loss.backward()
            clip_grad_norm_(self.q_network.parameters(), 1.0, norm_type=1)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / train_epochs

    def get_action_and_mutant(self, epsilon):
        """Return an action and the resulting mutant."""
        state_tensor = torch.FloatTensor([self.state.ravel()])
        prediction = self.calculate_next_q_values(state_tensor).detach().numpy()
        prediction = prediction.reshape((len(self.alphabet), self.seq_len))
        moves = renormalize_moves(self.state, prediction)
        # make action
        if moves.sum() > 0:
            p = random.random()
            action = sample_random(moves) if p < epsilon else sample_greedy(moves)
        else:
            # sometimes initialization of network causes prediction of all zeros,
            # causing moves of all zeros
            action = make_random_action(self.state.shape)
        # get next state (mutant)
        mutant = construct_mutant_from_sample(action, self.state)
        self.state = mutant

        return action, mutant

    def pick_action(self):
        """Pick an action.

        Generates a new string representing the state, along with its associated reward.
        """
        eps = max(
            self.epsilon_min, (0.5 - self.model.cost / (self.batch_size * self.rounds)),
        )
        state = self.state.copy()
        action, new_state = self.get_action_and_mutant(eps)
        new_state_string = one_hot_to_string(new_state, self.alphabet)
        reward = self.model.get_fitness(new_state_string)
        if not new_state_string in self.model.measured_sequences:
            if reward >= self.best_fitness:
                state_tensor = torch.FloatTensor([self.state.ravel()])
                prediction = self.calculate_next_q_values(state_tensor).detach().numpy()
                prediction = prediction.reshape((len(self.alphabet), self.seq_len))
                self.top_sequence.append((reward, new_state, self.model.cost))
            self.best_fitness = max(self.best_fitness, reward)
            self.memory.store(state.ravel(), action.ravel(), reward, new_state.ravel())
        if (
            self.model.cost > 0
            and self.model.cost % self.batch_size == 0
            and len(self.memory) >= self.batch_size
        ):
            self.train_actor(self.train_epochs)
        self.num_actions += 1
        return new_state_string, reward

    def propose_samples(self):
        """Propose `batch_size` samples."""
        if self.num_actions == 0:
            # indicates model was reset
            self.initialize_data_structures()
        seq_preds = set()
        prev_cost = copy.deepcopy(self.model.cost)
        total_evals = 0
        while (self.model.cost - prev_cost < self.batch_size) and (
            total_evals < self.batch_size * self.virtual_screen
        ):
            new_state_string, pred = self.pick_action()
            seq_preds.add((new_state_string, pred))
            total_evals += 1
        if len(seq_preds) < self.batch_size:
            random_sequences = generate_random_sequences(
                self.seq_len, self.batch_size - len(seq_preds), self.alphabet
            )
            seq_preds.update(random_sequences)
        samples, preds = zip(*seq_preds)
        return list(samples), list(preds)
