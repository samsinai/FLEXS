"""DyNA-PPO environment module."""

import editdistance
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import flexs
import flexs.utils.sequence_utils as s_utils


class DynaPPOEnvironment(py_environment.PyEnvironment):  # pylint: disable=W0223
    """DyNA-PPO environment based on TF-Agents."""

    def __init__(  # pylint: disable=W0231
        self, alphabet: str, seq_length: int, model: flexs.Model
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

        self.seq_length = seq_length
        self.partial_seq_len = 0
        self.state = np.zeros((seq_length, len(alphabet) + 1), dtype="float32")
        self.state[np.arange(len(self.state)), -1] = 1

        # model/model/measurements
        self.model = model
        self.previous_fitness = -float("inf")

        # sequence
        self.all_seqs = {}
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
            shape=(self.seq_length, len(self.alphabet) + 1),
            dtype=np.float32,
            minimum=0,
            maximum=1,
            name="observation",
        )
        self._time_step_spec = ts.time_step_spec(self._observation_spec)

    def _reset(self):
        self.partial_seq_len = 0
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

    def get_cached_fitness(self, seq):
        return self.all_seqs[seq]

    def _step(self, action):
        """Progress the agent one step in the environment.
        """

        self.state[self.partial_seq_len, -1] = 0
        self.state[self.partial_seq_len, action] = 1
        self.partial_seq_len += 1

        # We have not generated the last residue in the sequence, so continue
        if self.partial_seq_len < self.seq_length - 1:
            return ts.transition(self.state, 0)

        # If sequence is of full length, score the sequence and end the episode
        # We need to take off the column in the matrix (-1) representing the mask token
        complete_sequence = s_utils.one_hot_to_string(self.state[:, :-1], self.alphabet)
        fitness = self.model.get_fitness([complete_sequence]).item()
        self.all_seqs[complete_sequence] = fitness

        reward = fitness - self.lam * self.sequence_density(complete_sequence)
        return ts.termination(self.state, reward)
