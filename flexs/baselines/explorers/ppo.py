"""PPO explorer."""

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
from flexs.baselines.explorers.environments.ppo import PPOEnvironment as PPOEnv
from flexs.utils.sequence_utils import one_hot_to_string

import pandas as pd
from typing import Set, Tuple, Type


class PPO(flexs.Explorer):
    """Explorer for PPO."""

    def __init__(
        self,
        model: flexs.Model,
        landscape: flexs.Landscape,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        log_file: str = None,
    ):
        """Explorer which uses PPO.

        The algorithm is:
            for N experiment rounds
                collect samples with policy
                train policy on samples

        Attributes:
            meas_seqs: All measured sequences.
            meas_seqs_it: Iterator through `meas_seqs`.
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

        self.agent_sequences_data = None
        self.agent_sequences_data_iter = 0

        self.tf_env = None
        self.agent = None

        self.explorer_type = "PPO_Agent"

    def initialize_env(self):
        """Initialize TF-Agents environment."""
        env = PPOEnv(
            alphabet=self.alphabet,
            starting_seq=self.agent_sequences_data.iloc[0]["sequence"],
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
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

            self.agent_sequences_data_iter = (self.agent_sequences_data_iter + 1) % len(
                self.agent_sequences_data
            )
            self.tf_env.pyenv.envs[0].seq = self.agent_sequences_data.iloc[
                self.agent_sequences_data_iter
            ]["sequence"]

    def get_replay_buffer_and_driver(
        self, new_sequence_bucket: Set[str]
    ) -> Tuple[
        Type[tf_uniform_replay_buffer.TFUniformReplayBuffer],
        Type[dynamic_episode_driver.DynamicEpisodeDriver],
    ]:
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
                partial(self.add_last_seq_in_trajectory, new_seqs=new_sequence_bucket),
                tf_metrics.NumberOfEpisodes(),
                tf_metrics.EnvironmentSteps(),
            ],
            num_episodes=1,
        )

        return replay_buffer, collect_driver

    def pretrain_agent(self, measured_sequences_data: pd.DataFrame):
        """Pretrain the agent.

        Because of the budget constraint, we can only pretrain the agent on so many
        sequences (this number is currently set to `self.model_queries_per_batch / 2`).
        """
        self.agent_sequences_data = measured_sequences_data[
            ["sequence", "model_score"]
        ].copy()
        self.agent_sequences_data = self.agent_sequences_data.sort_values(
            by="model_score", ascending=False
        )

        self.initialize_env()
        self.initialize_agent()

        max_model_cost = self.model_queries_per_batch / 2

        seen_seqs = set(self.agent_sequences_data["sequence"])
        proposed_seqs = set()

        replay_buffer, collect_driver = self.get_replay_buffer_and_driver(
            new_sequence_bucket=proposed_seqs
        )

        print(f"Evaluating {max_model_cost} new sequences for pretraining...")
        previous_model_cost = self.model.cost
        while (self.model.cost - previous_model_cost) < max_model_cost:
            print(f"Total evals: {self.model.cost - previous_model_cost}")

            # generate new sequences
            for _ in range(self.sequences_batch_size):
                collect_driver.run()
                if (self.model.cost - previous_model_cost) >= max_model_cost:
                    break

            # get proposed sequences which have not already been measured
            # (since the landscape is not updating)
            new_proposed_seqs = list(proposed_seqs.difference(seen_seqs))

            # add new sequences to agent_sequences_data and sort
            self.agent_sequences_data = self.agent_sequences_data.append(
                pd.DataFrame(
                    {
                        "sequence": new_proposed_seqs,
                        "model_score": self.model.get_fitness(new_proposed_seqs),
                    }
                )
            )
            self.agent_sequences_data = self.agent_sequences_data.sort_values(
                by="model_score", ascending=False
            )

            print(f"Number of measured sequences: {len(self.agent_sequences_data)}")

            # add proposed sequences to set of all sequences
            seen_seqs.update(proposed_seqs)

            # reset iterator
            self.agent_sequences_data_iter = 0

            # reset proposed sequences
            proposed_seqs.clear()

            # train from the agent's trajectories
            trajectories = replay_buffer.gather_all()
            self.agent.train(experience=trajectories)
            replay_buffer.clear()

    def propose_sequences(self, measured_sequences_data: pd.DataFrame):
        """Propose `batch_size` samples."""

        if measured_sequences_data["round"].max() == 0:
            self.pretrain_agent(measured_sequences_data)

        seen_seqs = set(self.agent_sequences_data["sequence"])
        proposed_seqs = set()

        replay_buffer, collect_driver = self.get_replay_buffer_and_driver(
            new_sequence_bucket=proposed_seqs
        )

        # reset counter?
        self.agent_sequences_data_iter = 0

        # since we used part of the total budget for pretraining, amortize this cost
        effective_budget = (
            self.rounds * self.model_queries_per_batch
            - self.model_queries_per_batch / 2
        ) / self.rounds

        previous_model_cost = self.model.cost
        while (self.model.cost - previous_model_cost) < effective_budget:
            collect_driver.run()
            if (self.model.cost - previous_model_cost) % 500 == 0:
                print(self.model.cost - previous_model_cost)
            # we've looped over, found nothing new
            if self.agent_sequences_data_iter == 0:
                break

        proposed_seqs = list(proposed_seqs.difference(seen_seqs))
        measured_proposed_seqs = pd.DataFrame(
            {
                "sequence": proposed_seqs,
                "model_score": self.model.get_fitness(proposed_seqs),
            }
        )
        measured_proposed_seqs = measured_proposed_seqs.sort_values(
            by="model_score", ascending=False
        )

        self.agent_sequences_data = pd.concat(
            [self.agent_sequences_data, measured_proposed_seqs]
        )
        self.agent_sequences_data = self.agent_sequences_data.sort_values(
            by="model_score", ascending=False
        )

        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()

        return (
            measured_proposed_seqs["sequence"].to_numpy()[: self.sequences_batch_size],
            measured_proposed_seqs["model_score"].to_numpy()[
                : self.sequences_batch_size
            ],
        )
