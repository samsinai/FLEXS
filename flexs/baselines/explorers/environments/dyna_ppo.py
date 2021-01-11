"""DyNA-PPO environment module."""
import editdistance
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import nest_utils

import flexs
from flexs.utils import sequence_utils as s_utils


class DynaPPOEnvironment(py_environment.PyEnvironment):  # pylint: disable=W0223
    """DyNA-PPO environment based on TF-Agents."""

    def __init__(  # pylint: disable=W0231
        self,
        alphabet: str,
        seq_length: int,
        model: flexs.Model,
        landscape: flexs.Landscape,
        batch_size: int,
    ):
        """
        Initialize DyNA-PPO agent environment.

        Based on this tutorial:
        https://www.mikulskibartosz.name/how-to-create-an-environment-for-a-tensorflow-agent

        Args:
            alphabet: Usually UCGA.
            starting_seq: When initializing the environment,
                the sequence which is initially mutated.
            model: Landscape or model which evaluates
                each sequence.
            landscape: True fitness landscape.
            batch_size: Number of epsisodes to batch together and run in parallel.

        """
        self.alphabet = alphabet
        self._batch_size = batch_size

        self.seq_length = seq_length
        self.partial_seq_len = 0
        self.states = np.zeros(
            (batch_size, seq_length, len(alphabet) + 1), dtype="float32"
        )
        self.states[:, np.arange(seq_length), -1] = 1

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
            shape=(),
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
        self.states[:, :, :] = 0
        self.states[:, np.arange(self.seq_length), -1] = 1
        return nest_utils.stack_nested_arrays(
            [ts.restart(seq_state) for seq_state in self.states]
        )

    def batched(self):
        """Tf-agents function that says that this env returns batches of timesteps."""
        return True

    @property
    def batch_size(self):
        """Tf-agents property that return env batch size."""
        return self._batch_size

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
        """Get cached sequence fitness computed in previous episodes."""
        return self.all_seqs[seq]

    def set_fitness_model_to_gt(self, fitness_model_is_gt):
        """
        Set the fitness model to the ground truth landscape or to the model.

        Call with `True` when doing an experiment-based training round
        and call with `False` when doing a model-based training round.
        """
        self.fitness_model_is_gt = fitness_model_is_gt

    def _step(self, actions):
        """Progress the agent one step in the environment."""
        actions = actions.flatten()
        self.states[:, self.partial_seq_len, -1] = 0
        self.states[np.arange(self._batch_size), self.partial_seq_len, actions] = 1
        self.partial_seq_len += 1

        # We have not generated the last residue in the sequence, so continue
        if self.partial_seq_len < self.seq_length - 1:
            return nest_utils.stack_nested_arrays(
                [ts.transition(seq_state, 0) for seq_state in self.states]
            )

        # If sequence is of full length, score the sequence and end the episode
        # We need to take off the column in the matrix (-1) representing the mask token
        complete_sequences = [
            s_utils.one_hot_to_string(seq_state[:, :-1], self.alphabet)
            for seq_state in self.states
        ]
        if self.fitness_model_is_gt:
            fitnesses = self.landscape.get_fitness(complete_sequences)
        else:
            fitnesses = self.model.get_fitness(complete_sequences)
        self.all_seqs.update(zip(complete_sequences, fitnesses))

        # Reward = fitness - lambda * sequence density
        rewards = np.array(
            [
                f - self.lam * self.sequence_density(seq)
                for seq, f in zip(complete_sequences, fitnesses)
            ]
        )
        return nest_utils.stack_nested_arrays(
            [ts.termination(seq_state, r) for seq_state, r in zip(self.states, rewards)]
        )


class DynaPPOEnvironmentMutative(py_environment.PyEnvironment):  # pylint: disable=W0223
    """
    DyNA-PPO environment based on TF-Agents.

    Note that unlike the other DynaPPO environment, this one is mutative rather than
    constructive.
    """

    def __init__(  # pylint: disable=W0231
        self,
        alphabet: str,
        starting_seq: str,
        model: flexs.Model,
        landscape: flexs.Landscape,
        max_num_steps: int,
    ):
        """
        Initialize DyNA-PPO agent environment.

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
        self.alphabet = alphabet

        # model/model/measurements
        self.model = model
        self.landscape = landscape
        self.fitness_model_is_gt = False
        self.previous_fitness = -float("inf")

        self.seq = starting_seq
        self._state = {
            "sequence": s_utils.string_to_one_hot(self.seq, self.alphabet).astype(
                np.float32
            ),
            "fitness": self.model.get_fitness([starting_seq]).astype(np.float32),
        }
        self.episode_seqs = set()  # the sequences seen in the current episode
        self.all_seqs = {}
        self.measured_sequences = {}

        self.lam = 0.1

        # tf_agents environment
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
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
            "fitness": array_spec.BoundedArraySpec(
                shape=(1,), minimum=0, maximum=1, dtype=np.float32
            ),
        }

        self.num_steps = 0
        self.max_num_steps = max_num_steps

    def _reset(self):
        self.previous_fitness = -float("inf")
        self._state = {
            "sequence": s_utils.string_to_one_hot(self.seq, self.alphabet).astype(
                np.float32
            ),
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
            self._state["fitness"] = self.landscape.get_fitness([state_string]).astype(
                np.float32
            )
        else:
            self._state["fitness"] = self.model.get_fitness([state_string]).astype(
                np.float32
            )
        self.all_seqs[state_string] = self._state["fitness"].item()

        reward = self._state["fitness"].item() - self.lam * self.sequence_density(
            state_string
        )

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
