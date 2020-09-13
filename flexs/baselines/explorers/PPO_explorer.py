"""PPO explorer."""

import collections
from functools import partial

import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import flexs
from flexs.baselines.explorers.environments.PPO_environment import (
    PPOEnvironment as PPOEnv,
)
from flexs.baselines.explorers.base_explorer import Base_explorer
from flexs.utils.sequence_utils import one_hot_to_string


class PPO(flexs.Explorer):
    """Explorer for PPO."""

    def __init__(
        self,
        model,
        landscape,
        rounds,
        sequences_batch_size,
        model_queries_per_batch,
        starting_sequence,
        alphabet,
        batch_size,
        log_file=None,
    ):
        """Explorer which uses PPO.

        The algorithm is:
            for N experiment rounds
                collect samples with policy
                train policy on samples

        Attributes:
            meas_seqs: All measured sequences.
            meas_seqs_it: Iterator through `meas_seqs`.
            top_seqs: Top measured sequences by fitness score.
            top_seqs_it: Iterator through `top_seqs`.
            has_pretrained_agent: Whether or not the agent has been trained already.
            original_horizon: Total number of rounds. Used to compute proposal
                budget and distribute this budget between rounds.
            tf_env: TF-Agents environment in which to run the explorer.
            agent: Decision-making agent.
        """

        name = f"PPO_Agent"

        super().__init__(
            model,
            landscape,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        self.alphabet = alphabet

        self.meas_seqs = None
        self.meas_seqs_it = 0
        self.top_seqs = None
        self.top_seqs_it = None

        self.has_pretrained_agent = None
        self.original_horizon = None

        self.tf_env = None
        self.agent = None

        self.explorer_type = "PPO_Agent"

    def reset(self):
        """Reset the explorer."""
        self.meas_seqs = []
        self.meas_seqs_it = 0

        self.top_seqs = collections.deque(maxlen=self.sequences_batch_size)
        self.top_seqs_it = 0

        self.has_pretrained_agent = False
        self.batches = {-1: ""}

        self.original_horizon = None

    def reset_measured_seqs(self):
        """Reset the measured sequences."""
        measured_seqs = [
            (self.model.get_fitness(seq), seq, self.model.cost)
            for seq in self.model.measured_sequences
        ]
        measured_seqs = sorted(measured_seqs, key=lambda x: x[0], reverse=True)

        self.top_seqs = collections.deque(
            measured_seqs, maxlen=self.sequences_batch_size
        )
        self.meas_seqs = measured_seqs

    def initialize_env(self):
        """Initialize TF-Agents environment."""
        env = PPOEnv(
            alphabet=self.alphabet,
            starting_seq=self.meas_seqs[0][1],
            landscape=self.model,
            max_num_steps=self.model_queries_per_batch,
        )

        validate_py_environment(env, episodes=1)

        self.tf_env = tf_py_environment.TFPyEnvironment(env)

    def initialize_agent(self):
        """Initialize agent."""
        actor_fc_layers = [128]
        value_fc_layers = [128]

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=actor_fc_layers,
        )
        value_net = value_network.ValueNetwork(
            self.tf_env.observation_spec(), fc_layer_params=value_fc_layers
        )

        num_epochs = 10
        agent = ppo_agent.PPOAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5),
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=num_epochs,
            summarize_grads_and_vars=False,
        )
        agent.initialize()

        self.agent = agent

    def add_last_seq_in_trajectory(self, experience, new_seqs):
        """Add the last sequence in an episode's trajectory.

        Given a trajectory object, checks if the object is the last in the trajectory.
        Since the environment ends the episode when the score is non-increasing, it
        adds the associated maximum-valued sequence to the batch.

        If the episode is ending, it changes the "current sequence" of the environment
        to the next one in `last_batch`, so that when the environment resets, mutants
        are generated from that new sequence.
        """
        if experience.is_boundary():
            seq = one_hot_to_string(experience.observation.numpy()[0], self.alphabet)
            new_seqs.add(seq)

            self.meas_seqs_it = (self.meas_seqs_it + 1) % len(self.meas_seqs)
            self.tf_env.pyenv.envs[0].seq = self.meas_seqs[self.meas_seqs_it][1]

    def pretrain_agent(self, measured_sequences):
        """Pretrain the agent.

        Because of the budget constraint, we can only pretrain the agent on so many
        sequences (this number is currently set to `self.sequences_batch_size * self.
        model_queries_per_batch / 2`).
        """
        measured_seqs = [
            (self.model.get_fitness(seq), seq, self.model.cost)
            for seq in measured_sequences["sequence"]
        ]
        measured_seqs = sorted(measured_seqs, key=lambda x: x[0], reverse=True)

        self.top_seqs = collections.deque(
            measured_seqs, maxlen=self.sequences_batch_size
        )
        self.meas_seqs = measured_seqs

        self.initialize_env()
        self.initialize_agent()

        batch_size = self.sequences_batch_size
        max_new_evals = self.sequences_batch_size * self.model_queries_per_batch / 2

        all_seqs = set(measured_sequences["sequence"])
        proposed_seqs = set()
        measured_seqs = []

        num_parallel_environments = 1
        env_steps_metric = tf_metrics.EnvironmentSteps()
        step_metrics = [tf_metrics.NumberOfEpisodes(), env_steps_metric]

        replay_buffer_capacity = 10001
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity,
        )

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            self.agent.collect_policy,
            observers=[
                replay_buffer.add_batch,
                partial(self.add_last_seq_in_trajectory, new_seqs=proposed_seqs),
            ]
            + step_metrics,
            num_episodes=1,
        )

        print(f"Evaluating {max_new_evals} new sequences for pretraining...")
        previous_evals = self.model.cost
        while (self.model.cost - previous_evals) < max_new_evals:
            print(f"Total evals: {self.model.cost - previous_evals}")

            # generate new sequences
            for _ in range(batch_size):
                collect_driver.run()
                if (self.model.cost - previous_evals) >= max_new_evals:
                    break

            # get proposed sequences which have not already been measured
            # (since the landscape is not updating)
            new_seqs = proposed_seqs.difference(all_seqs)

            # add new sequences to measured_sequences and sort
            self.meas_seqs += [
                (self.model.get_fitness(seq), seq, self.model.cost) for seq in new_seqs
            ]
            self.meas_seqs = sorted(self.meas_seqs, key=lambda x: x[0], reverse=True)

            print(f"Number of measured sequences: {len(self.meas_seqs)}")
            # if we have a new winner
            if len(self.top_seqs) == 0 or self.meas_seqs[0][0] > self.top_seqs[-1][0]:
                print("New top sequence:", self.meas_seqs[0])
                self.top_seqs.append(self.meas_seqs[0])

            # add proposed sequences to set of all sequences
            all_seqs.update(proposed_seqs)

            # reset counter
            self.meas_seqs_it = 0

            # reset proposed sequences
            proposed_seqs.clear()

            # train from the agent's trajectories
            trajectories = replay_buffer.gather_all()
            self.agent.train(experience=trajectories)
            replay_buffer.clear()

        self.has_pretrained_agent = True

    def propose_sequences(self, measured_sequences):
        """Propose `batch_size` samples."""
        if self.original_horizon is None:
            self.original_horizon = self.rounds

        if not self.has_pretrained_agent:
            self.pretrain_agent(measured_sequences)

        if len(self.meas_seqs) == 0:
            self.reset_measured_seqs()

        print("Proposing samples...")

        all_seqs = set(measured_sequences["sequence"])
        new_seqs = set()

        num_parallel_environments = 1

        replay_buffer_capacity = 10001
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity,
        )

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            self.agent.collect_policy,
            observers=[
                replay_buffer.add_batch,
                partial(self.add_last_seq_in_trajectory, new_seqs=new_seqs),
            ],
            num_episodes=1,
        )

        # reset counter?
        self.meas_seqs_it = 0

        # since we used part of the total budget for pretraining, amortize this cost
        effective_budget = (
            self.original_horizon
            * self.sequences_batch_size
            * self.model_queries_per_batch
            - (self.sequences_batch_size * self.model_queries_per_batch / 2)
        ) / self.original_horizon

        previous_evals = self.model.cost
        while (self.model.cost - previous_evals) < effective_budget:
            collect_driver.run()
            if (self.model.cost - previous_evals) % 500 == 0:
                print(self.model.cost - previous_evals)
            # we've looped over, found nothing new
            if self.meas_seqs_it == 0:
                break

        new_seqs = new_seqs.difference(all_seqs)

        # add new sequences to measured_sequences and sort
        new_meas_seqs = [
            (self.model.get_fitness(seq), seq, self.model.cost) for seq in new_seqs
        ]
        new_meas_seqs = sorted(new_meas_seqs, key=lambda x: x[0], reverse=True)
        self.meas_seqs += new_meas_seqs
        self.meas_seqs = sorted(self.meas_seqs, key=lambda x: x[0], reverse=True)

        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()

        return [s[1] for s in new_meas_seqs[: self.sequences_batch_size]], None
