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
        self,
        alphabet: str,
        seq_length: int,
        model: flexs.Model,
        landscape: flexs.Landscape,
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
        self.landscape = landscape
        self.fitness_model_is_gt = False
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

    def set_fitness_model_to_gt(self, fitness_model_is_gt):
        """
        Set the fitness model to the ground truth landscape or to the model.

        Call with `True` when doing an experiment-based training round
        and call with `False` when doing a model-based training round.
        """
        self.fitness_model_is_gt = fitness_model_is_gt

    def _step(self, action):
        """Progress the agent one step in the environment."""
        self.state[self.partial_seq_len, -1] = 0
        self.state[self.partial_seq_len, action] = 1
        self.partial_seq_len += 1

        # We have not generated the last residue in the sequence, so continue
        if self.partial_seq_len < self.seq_length - 1:
            return ts.transition(self.state, 0)

        # If sequence is of full length, score the sequence and end the episode
        # We need to take off the column in the matrix (-1) representing the mask token
        complete_sequence = s_utils.one_hot_to_string(self.state[:, :-1], self.alphabet)
        if self.fitness_model_is_gt:
            fitness = self.landscape.get_fitness([complete_sequence]).item()
        else:
            fitness = self.model.get_fitness([complete_sequence]).item()
        self.all_seqs[complete_sequence] = fitness

        reward = fitness - self.lam * self.sequence_density(complete_sequence)
        return ts.termination(self.state, reward)

class DynaPPOEnvironmentMutative(py_environment.PyEnvironment):  # pylint: disable=W0223
    """DyNA-PPO environment based on TF-Agents. Note that unlike the other DynaPPO environment, this one is mutative rather than constructive."""

    def __init__(  # pylint: disable=W0231
        self,
        alphabet: str,
        starting_seq: str,
        model: flexs.Model,
        landscape: flexs.Landscape,
        max_num_steps: int
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
        """
        # alphabet
        self.alphabet = alphabet
        
        # model/model/measurements
        self.model = model
        self.landscape = landscape
        self.fitness_model_is_gt = False
        self.previous_fitness = -float("inf")

        self.seq = starting_seq
        self._state = {
            "sequence": s_utils.string_to_one_hot(self.seq, self.alphabet).astype(np.float32),
            "fitness": self.model.get_fitness([starting_seq]).astype(np.float32),
        }
        self.episode_seqs = set()  # the sequences seen in the current episode
        self.all_seqs = {}
        self.measured_sequences = {}

        self.lam = 0.1

        # tf_agents environment
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,),
            dtype=np.integer,
            minimum=0,
            maximum=len(self.seq) * len(self.alphabet) - 1,
            name="action",
        )
        self._observation_spec = {
            "sequence": array_spec.BoundedArraySpec(
                shape=(len(self.seq), len(self.alphabet)),
                dtype=np.float32,
                minimum=0,
                maximum=1,
            ),
            "fitness": array_spec.ArraySpec(shape=(1,), dtype=np.float32),
        }

        self.num_steps = 0
        self.max_num_steps = max_num_steps

    def _reset(self):
        self.previous_fitness = -float("inf")
        self._state = {
            "sequence": s_utils.string_to_one_hot(self.seq, self.alphabet).astype(np.float32),
            "fitness": self.model.get_fitness([self.seq]).astype(np.float32),
        }
        self.episode_seqs = set()
        self.num_steps = 0
        return ts.restart(self._state)

    def action_spec(self):
        """Define agent actions."""
        return self._action_spec

    def observation_spec(self):
        """Define environment observations."""
        return self._observation_spec

    def get_state_string(self):
        """Get sequence representing current state."""
        return s_utils.one_hot_to_string(self._state["sequence"], self.alphabet)

    def sequence_density(self, seq):
        """Get average distance to `seq` out of all observed sequences."""
        dens = 0
        dist_radius = 2
        for s in self.all_seqs:
            dist = int(editdistance.eval(s, seq))
            if dist != 0 and dist <= dist_radius:
                dens += self.all_seqs[s] / dist
        return dens

    def set_fitness_model_to_gt(self, fitness_model_is_gt):
        """
        Set the fitness model to the ground truth landscape or to the model.

        Call with `True` when doing an experiment-based training round
        and call with `False` when doing a model-based training round.
        """
        self.fitness_model_is_gt = fitness_model_is_gt

    def _step(self, action):
        """Progress the agent one step in the environment.

        The agent moves until the reward is decreasing. The number of sequences that
        can be evaluated at each episode is capped to `self.max_num_steps`.
        """

        # if we've exceeded the maximum number of steps, terminate
        if self.num_steps >= self.max_num_steps:
            return ts.termination(self._state, 0)

        # `action` is an integer representing which residue to mutate to 1
        # along the flattened one-hot representation of the sequence
        pos = action // len(self.alphabet)
        res = action % len(self.alphabet)
        self.num_steps += 1

        # if we are trying to modify the sequence with a no-op, then stop
        if self._state["sequence"][pos, res] == 1:
            return ts.termination(self._state, 0)

        self._state["sequence"][pos] = 0
        self._state["sequence"][pos, res] = 1
        state_string = s_utils.one_hot_to_string(self._state["sequence"], self.alphabet)

        if self.fitness_model_is_gt:
            self._state["fitness"] = self.landscape.get_fitness([state_string]).astype(np.float32)
        else:
            self._state["fitness"] = self.model.get_fitness([state_string]).astype(np.float32)
        self.all_seqs[state_string] = self._state["fitness"].item()

        reward = self._state["fitness"].item() - self.lam * self.sequence_density(state_string)

        # if we have seen the sequence this episode,
        # terminate episode and punish
        # (to prevent going in loops)
        if state_string in self.episode_seqs:
            return ts.termination(self._state, -1)
        self.episode_seqs.add(state_string)

        # if the reward is not increasing, then terminate
        if reward < self.previous_fitness:
            return ts.termination(self._state, reward=reward)

        self.previous_fitness = reward
        return ts.transition(self._state, reward=reward)
