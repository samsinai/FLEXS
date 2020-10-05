"""DQN explorer."""
import random
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim as optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

import flexs
from flexs.utils.replay_buffers import PrioritizedReplayBuffer
from flexs.utils.sequence_utils import (
    construct_mutant_from_sample,
    one_hot_to_string,
    string_to_one_hot,
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


class DQN(flexs.Explorer):
    """
    DQN explorer class.

    DQN Explorer implementation, based off
    https://colab.research.google.com/drive/1NsbSPn6jOcaJB_mp9TmkgQX7UrRIrTi0.

    Algorithm works as follows:
    for N experiment rounds
        collect samples with policy
        policy updates using Q network:
            Q(s, a) <- Q(s, a) + alpha * (R(s, a) + gamma * max Q(s, a) - Q(s, a))
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
        memory_size: int = 100000,
        train_epochs: int = 20,
        gamma: float = 0.9,
        device: str = "cpu",
    ):
        """
        Args:
            memory_size: Size of agent memory.
            gamma: Discount factor.
        """
        name = "DQN_Explorer"
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
        self.state = string_to_one_hot(self.starting_sequence, self.alphabet)
        self.seq_len = len(self.starting_sequence)
        self.q_network = build_q_network(self.seq_len, len(self.alphabet), self.device)
        self.q_network.eval()
        self.memory = PrioritizedReplayBuffer(
            len(self.alphabet) * self.seq_len,
            self.memory_size,
            self.sequences_batch_size,
            0.6,
        )

    def sample(self):
        """Sample a random `batch_size` subset of the memory."""
        indices = np.random.choice(len(self.memory), self.sequences_batch_size)
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
        prediction = prediction.reshape((self.seq_len, len(self.alphabet)))

        # Ensure that staying in place gives no reward
        zero_current_state = (self.state - 1) * (-1)
        moves = np.multiply(prediction, zero_current_state)

        # make action
        if moves.sum() > 0:
            # Sample a random action
            if random.random() < epsilon:
                i, j = moves.shape
                non_zero_moves = np.nonzero(moves)
                num_moves = len(non_zero_moves)
                num_pos = len(non_zero_moves[0])
                if num_moves != 0 and num_pos != 0:
                    rand_arg = random.choice(
                        [
                            [non_zero_moves[alph][pos] for alph in range(num_moves)]
                            for pos in range(num_pos)
                        ]
                    )
                else:
                    rand_arg = [random.randint(0, i - 1), random.randint(0, j - 1)]
                y = rand_arg[1]
                x = rand_arg[0]
                action = np.zeros((i, j))
                action[x][y] = moves[x][y]

            # Choose the greedy action
            else:
                i, j = moves.shape
                max_arg = np.argmax(moves)
                y = max_arg % j
                x = int(max_arg / j)
                action = np.zeros((i, j))
                action[x][y] = moves[x][y]

        else:
            # sometimes initialization of network causes prediction of all zeros,
            # causing moves of all zeros
            action = np.zeros(self.state.shape)
            action[
                np.random.randint(self.state.shape[0]),
                np.random.randint(self.state.shape[1]),
            ] = 1

        # get next state (mutant)
        mutant = construct_mutant_from_sample(action, self.state)
        self.state = mutant

        return action, mutant

    def pick_action(self, all_measured_seqs):
        """
        Pick an action.

        Generates a new string representing the state, along with its associated reward.
        """
        eps = max(
            self.epsilon_min,
            (0.5 - self.model.cost / (self.sequences_batch_size * self.rounds)),
        )
        state = self.state.copy()
        action, new_state = self.get_action_and_mutant(eps)
        new_state_string = one_hot_to_string(new_state, self.alphabet)
        reward = self.model.get_fitness([new_state_string]).item()
        if new_state_string not in all_measured_seqs:
            if reward >= self.best_fitness:
                state_tensor = torch.FloatTensor([self.state.ravel()])
                prediction = self.calculate_next_q_values(state_tensor).detach().numpy()
                prediction = prediction.reshape((len(self.alphabet), self.seq_len))
                self.top_sequence.append((reward, new_state, self.model.cost))
            self.best_fitness = max(self.best_fitness, reward)
            self.memory.store(state.ravel(), action.ravel(), reward, new_state.ravel())
        if (
            self.model.cost > 0
            and self.model.cost % self.sequences_batch_size == 0
            and len(self.memory) >= self.sequences_batch_size
        ):
            self.train_actor(self.train_epochs)
        self.num_actions += 1
        return new_state_string, reward

    def propose_sequences(
        self, measured_sequences_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        if self.num_actions == 0:
            # indicates model was reset
            self.initialize_data_structures()

        all_measured_seqs = set(measured_sequences_data["sequence"].values)
        sequences = {}

        prev_cost = self.model.cost
        while self.model.cost - prev_cost < self.model_queries_per_batch:
            new_state_string, pred = self.pick_action(all_measured_seqs)
            all_measured_seqs.add(new_state_string)
            sequences[new_state_string] = pred

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        return new_seqs[sorted_order], preds[sorted_order]
