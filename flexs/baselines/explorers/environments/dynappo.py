"""DyNA-PPO environment module."""
import os
import sys

import editdistance
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import flexs
from flexs.utils import sequence_utils as s_utils

from typing import Callable, List
from flexs.landscape import SEQUENCES_TYPE
from flexs.utils.sequence_utils import (
    construct_mutant_from_sample,
    one_hot_to_string,
    string_to_one_hot,
)


class DynaPPOEnvironment(py_environment.PyEnvironment):  # pylint: disable=W0223
    """DyNA-PPO environment based on TF-Agents."""

    def __init__(  # pylint: disable=W0231
        self,
        alphabet: str,
        starting_seq: str,
        landscape: flexs.Landscape,
        max_num_steps: int,
        get_fitness_ensemble: Callable[[SEQUENCES_TYPE], np.ndarray],
        give_oracle_reward: bool = False,
    ):
        """Initialize DyNA-PPO agent environment.

        Based on this tutorial:
        https://www.mikulskibartosz.name/how-to-create-an-environment-for-a-tensorflow-agent

        Args:
            alphabet: Usually UCGA.
            starting_seq: When initializing the environment,
                the sequence which is initially mutated.
            landscape: Landscape or model which evaluates
                each sequence.
            max_num_steps: Maximum number of steps before
                episode is forced to terminate. Usually the
                `model_queries_per_batch`.
            get_fitness_ensemble: Ensemble model fitness function.
            give_oracle_reward: Whether or not to give reward based
                on oracle or on ensemble model.
        """
        # alphabet
        self.alphabet = alphabet
        self.alphabet_len = len(self.alphabet)

        # landscape/model/measurements
        self.landscape = landscape
        self.previous_fitness = -float("inf")
        self.get_fitness_ensemble = get_fitness_ensemble
        self.give_oracle_reward = give_oracle_reward

        # sequence
        self.seq = starting_seq
        self.seq_len = len(self.seq)
        self._state = string_to_one_hot(self.seq, self.alphabet)
        self.all_seqs = {}
        self.episode_seqs = {}
        self.lam = 0.1

        # tf_agents environment
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1, 2), dtype=np.float32, minimum=0, maximum=1, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.seq_len, self.alphabet_len),
            dtype=np.float32,
            minimum=0,
            maximum=1,
            name="observation",
        )
        self._time_step_spec = ts.time_step_spec(self._observation_spec)
        self._episode_ended = False

        self.max_num_steps = max_num_steps
        self.num_steps = 0

    def _reset(self):
        self.previous_fitness = -float("inf")
        self._state = s_utils.string_to_one_hot(self.seq, self.alphabet)
        self.num_steps = 0
        self.episode_seqs = {}
        return ts.restart(np.array(self._state, dtype=np.float32))

    def time_step_spec(self):
        """Define time steps."""
        return self._time_step_spec

    def action_spec(self):
        """Define agent actions."""
        return self._action_spec

    def observation_spec(self):
        """Define environment observations."""
        return self._observation_spec

    def get_state_string(self):
        """Get sequence representing current state."""
        return one_hot_to_string(self._state, self.alphabet)

    def sequence_density(self, seq):
        """Get average distance to `seq` out of all observed sequences."""
        dens = 0
        dist_radius = 2
        for s in self.all_seqs:
            dist = int(editdistance.eval(s, seq))
            if dist != 0 and dist <= dist_radius:
                dens += self.all_seqs[s] / dist
        return dens

    def _step(self, action):
        """Progress the agent one step in the environment.

        The agent moves until the reward is decreasing. The number of sequences that
        can be evaluated at each episode is capped to `self.max_num_steps`.
        """
        if self.num_steps < self.max_num_steps:
            self.num_steps += 1
            action_one_hot = np.zeros((self.seq_len, self.alphabet_len))

            # if action is invalid,
            # terminate episode and punish
            if np.amax(action) >= 1 or np.amin(action) < 0:
                return ts.termination(np.array(self._state, dtype=np.float32), -1)

            x, y = action[0]
            x, y = int(self.seq_len * x), int(self.alphabet_len * y)
            action_one_hot[x, y] = 1

            # if we are trying to modify the sequence with a no-op, then stop
            if self._state[x, y] == 1:
                self._episode_ended = True

                # if the next best move is to stay at the current state,
                # then give it a small reward
                return ts.termination(np.array(self._state, dtype=np.float32), 1)

            self._state = construct_mutant_from_sample(
                action_one_hot, self._state
            )
            state_string = one_hot_to_string(self._state, self.alphabet)

            # if we have seen the sequence this episode,
            # terminate episode and punish
            # (to prevent going in loops)
            if state_string in self.episode_seqs:
                return ts.termination(np.array(self._state, dtype=np.float32), -1)
            self.episode_seqs[state_string] = 1

            if self.give_oracle_reward:
                reward = self.landscape.get_fitness([state_string])[0]
            else:
                reward = self.get_fitness_ensemble([state_string])[0]

            reward = reward - self.lam * self.sequence_density(state_string)

            # if the reward is not increasing, then terminate
            if reward < self.previous_fitness:
                if state_string not in self.all_seqs:
                    self.all_seqs[state_string] = 0
                self.all_seqs[state_string] += 1
                return ts.termination(
                    np.array(self._state, dtype=np.float32), reward=reward
                )

            self.previous_fitness = reward
            return ts.transition(np.array(self._state, dtype=np.float32), reward=reward)

        # if we've exceeded the maximum number of steps, terminate
        self._episode_ended = True
        return ts.termination(np.array(self._state, dtype=np.float32), 0)
