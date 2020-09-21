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
from flexs.model import SEQUENCES_TYPE
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
        model: flexs.Model,
        max_num_steps: int,
        get_fitness_ensemble: Callable[[SEQUENCES_TYPE], np.ndarray],
    ):
        """Initialize DyNA-PPO agent environment.

        Based on this tutorial:
        https://www.mikulskibartosz.name/how-to-create-an-environment-for-a-tensorflow-agent

        Args:
            alphabet: Usually UCGA.
            starting_seq: When initializing the environment,
                the sequence which is initially mutated.
            model: Landscape or model which evaluates
                each sequence.
            max_num_steps: Maximum number of steps before
                episode is forced to terminate. Usually the
                `model_queries_per_batch`.
            get_fitness_ensemble: Ensemble model fitness function.
            give_oracle_reward: Whether or not to give reward based
                on oracle or on ensemble model.
        """

        self.alphabet = alphabet

        self.current_seq_len = 0
        self.state = np.zeros((len(starting_seq), len(alphabet + 1)))
        self.state[np.arange(len(starting_seq)), -1] = 1

        # model/model/measurements
        self.model = model
        self.previous_fitness = -float("inf")
        self.get_fitness_ensemble = get_fitness_ensemble
        self.give_oracle_reward = False

        # sequence
        self.seq = starting_seq
        self.seq_len = len(self.seq)
        self._state = string_to_one_hot(self.seq, self.alphabet)
        self.all_seqs = {}
        self.episode_seqs = {}
        self.lam = 0.1

        # tf_agents environment
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,),
            dtype=np.integer,
            minimum=0,
            maximum=len(self.alphabet) - 1,
            name="action",
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.seq_len, self.alphabet_len),
            dtype=np.float32,
            minimum=0,
            maximum=1,
            name="observation",
        )
        self._time_step_spec = ts.time_step_spec(self._observation_spec)

    def _reset(self):
        self.current_seq_len = 0
        self.state = np.zeros_like(self.state)
        self.state[np.arange(len(self.state)), -1] = 1
        return ts.restart(self.state)

    def time_step_spec(self):
        """Define time steps."""
        return self._time_step_spec

    def action_spec(self):
        """Define agent actions."""
        return self._action_spec

    def observation_spec(self):
        """Define environment observations."""
        return self._observation_spec

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
        """

        self.state[self.current_seq_len, -1] = 0
        self.state[self.current_seq_len, action] = 1
        self.current_seq_len += 1

        if self.current_seq_len < self.seq_length:
            return ts.transition(self.state, 0)

        complete_sequence = one_hot_to_string(self.state[:, :-1], self.alphabet)
        if self.give_oracle_reward:
            reward = self.model.get_fitness([complete_sequence]).item()
        else:
            reward = self.get_fitness_ensemble([complete_sequence]).item()

        reward = reward - self.lam * self.sequence_density(self.current_sequence)

        return ts.termination(self.state, reward)
