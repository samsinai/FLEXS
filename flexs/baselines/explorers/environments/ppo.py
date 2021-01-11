"""PPO environment module."""
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import flexs
from flexs.utils.sequence_utils import one_hot_to_string, string_to_one_hot


class PPOEnvironment(py_environment.PyEnvironment):  # pylint: disable=W0223
    """PPO environment based on TF-Agents."""

    def __init__(
        self,
        alphabet: str,
        starting_seq: str,
        model: flexs.Model,
        max_num_steps: int,
    ):  # pylint: disable=W0231
        """
        Initialize PPO agent environment.

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
        self.previous_fitness = -float("inf")

        # sequence
        self.seq = starting_seq
        self._state = {
            "sequence": string_to_one_hot(self.seq, self.alphabet).astype(np.float32),
            "fitness": self.model.get_fitness([starting_seq]).astype(np.float32),
        }
        self.episode_seqs = set()  # the sequences seen in the current episode
        self.measured_sequences = {}

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
                shape=(1,), minimum=1, maximum=1, dtype=np.float32
            ),
        }
        self._time_step_spec = ts.time_step_spec(self._observation_spec)

        self.num_steps = 0
        self.max_num_steps = max_num_steps

        validate_py_environment(self, episodes=1)

    def _reset(self):
        self.previous_fitness = -float("inf")
        self._state = {
            "sequence": string_to_one_hot(self.seq, self.alphabet).astype(np.float32),
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
        return one_hot_to_string(self._state["sequence"], self.alphabet)

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
        state_string = one_hot_to_string(self._state["sequence"], self.alphabet)
        self._state["fitness"] = self.model.get_fitness([state_string]).astype(
            np.float32
        )

        # if we have seen the sequence this episode,
        # terminate episode and punish
        # (to prevent going in loops)
        if state_string in self.episode_seqs:
            return ts.termination(self._state, -1)
        self.episode_seqs.add(state_string)

        # if the reward is not increasing, then terminate
        if self._state["fitness"] < self.previous_fitness:
            return ts.termination(self._state, reward=self._state["fitness"].item())

        self.previous_fitness = self._state["fitness"]
        return ts.transition(self._state, reward=self._state["fitness"].item())
