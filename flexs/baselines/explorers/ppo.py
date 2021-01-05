"""PPO explorer."""

from functools import partial
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import flexs
from flexs.baselines.explorers.environments.ppo import PPOEnvironment as PPOEnv
from flexs.utils.sequence_utils import one_hot_to_string


class PPO(flexs.Explorer):
    """
    Explorer which uses PPO.

    The algorithm is:
        for N experiment rounds
            collect samples with policy
            train policy on samples

    A simpler baseline than DyNAPPOMutative with similar performance.
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
    ):
        """Create PPO explorer."""
        super().__init__(
            model,
            "PPO_Agent",
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        self.alphabet = alphabet

        # Initialize tf_environment
        env = PPOEnv(
            alphabet=self.alphabet,
            starting_seq=starting_sequence,
            model=self.model,
            max_num_steps=self.model_queries_per_batch,
        )
        self.tf_env = tf_py_environment.TFPyEnvironment(env)

        encoder_layer = tf.keras.layers.Lambda(lambda obs: obs["sequence"])
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            preprocessing_combiner=encoder_layer,
            fc_layer_params=[128],
        )
        value_net = value_network.ValueNetwork(
            self.tf_env.observation_spec(),
            preprocessing_combiner=encoder_layer,
            fc_layer_params=[128],
        )

        # Create the PPO agent
        self.agent = ppo_agent.PPOAgent(
            time_step_spec=self.tf_env.time_step_spec(),
            action_spec=self.tf_env.action_spec(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=10,
            summarize_grads_and_vars=False,
        )
        self.agent.initialize()

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
            seq = one_hot_to_string(
                experience.observation["sequence"].numpy()[0], self.alphabet
            )
            new_seqs[seq] = experience.observation["fitness"].numpy().squeeze()

            top_fitness = max(new_seqs.values())
            top_sequences = [
                seq for seq, fitness in new_seqs.items() if fitness >= 0.9 * top_fitness
            ]
            if len(top_sequences) > 0:
                self.tf_env.pyenv.envs[0].seq = np.random.choice(top_sequences)
            else:
                self.tf_env.pyenv.envs[0].seq = np.random.choice(
                    [seq for seq, _ in new_seqs.items()]
                )

    def propose_sequences(
        self, measured_sequences_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        num_parallel_environments = 1
        replay_buffer_capacity = 10001
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity,
        )

        sequences = {}
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            self.agent.collect_policy,
            observers=[
                replay_buffer.add_batch,
                partial(self.add_last_seq_in_trajectory, new_seqs=sequences),
                tf_metrics.NumberOfEpisodes(),
                tf_metrics.EnvironmentSteps(),
            ],
            num_episodes=1,
        )

        previous_model_cost = self.model.cost
        while self.model.cost - previous_model_cost < self.model_queries_per_batch:
            collect_driver.run()

        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        sequences = {
            seq: fitness
            for seq, fitness in sequences.items()
            if seq not in set(measured_sequences_data["sequence"])
        }
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        return new_seqs[sorted_order], preds[sorted_order]
