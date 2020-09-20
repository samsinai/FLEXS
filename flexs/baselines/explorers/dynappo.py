"""DyNA-PPO explorer."""
from functools import partial

import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.gaussian_process
import sklearn.ensemble
import sklearn.tree
import sklearn.metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments.utils import validate_py_environment

import flexs
from flexs import baselines
from flexs.baselines.explorers.environments.dynappo import (
    DynaPPOEnvironment as DynaPPOEnv,
)
import flexs.utils.sequence_utils as s_utils
from typing import List, Set, Tuple, Type, Union


class DynaPPOEnsemble(baselines.models.AdaptiveEnsemble):
    def __init__(self, seq_len, r_squared_threshold, alphabet):
        super().__init__(
            models=[
                # FLEXS models
                baselines.models.GlobalEpistasisModel(seq_len, 100, alphabet),
                baselines.models.MLP(seq_len, 200, alphabet),
                baselines.models.CNN(seq_len, 32, 100, alphabet),
                # Sklearn models
                baselines.models.LinearRegression(alphabet),
                baselines.models.RandomForest(alphabet),
                baselines.models.SklearnRegressor(
                    sklearn.neighbors.NearestNeighbors(), alphabet, "nearest_neighbors",
                ),
                baselines.models.SklearnRegressor(
                    sklearn.linear_model.Lasso(), alphabet, "lasso"
                ),
                baselines.models.SklearnRegressor(
                    sklearn.linear_model.BayesianRidge(), alphabet, "bayesian_ridge",
                ),
                baselines.models.SklearnRegressor(
                    sklearn.gaussian_process.GaussianProcessRegressor(),
                    alphabet,
                    "gaussian_process",
                ),
                baselines.models.SklearnRegressor(
                    sklearn.ensemble.GradientBoostingRegressor(),
                    alphabet,
                    "gradient_boosting",
                ),
                baselines.models.SklearnRegressor(
                    sklearn.tree.ExtraTreeRegressor(), alphabet, "extra_trees"
                ),
            ]
        )

        self.r_squared_vals = None
        self.r_squared_threshold = r_squared_threshold

    def train(self, sequences, labels):
        (train_X, test_X, train_y, test_y,) = sklearn.model_selection.train_test_split(
            sequences, labels, test_size=0.25
        )

        for model in self.models:
            model.train(train_X, train_y)

        model_preds = np.stack(
            [model.get_fitness(test_X) for model in self.models], axis=0
        )

        self.r_squared_vals = np.array(
            [sklearn.metrics.r2_score(test_y, preds) for preds in model_preds]
        )

    def _fitness_function(self, sequences):
        scores = [
            model.get_fitness(sequences)
            for model, r_squared in zip(self.models, self.r_squared_vals)
            if r_squared > self.r_squared_threshold
        ]
        return np.mean(scores, axis=0)


class DynaPPO(flexs.Explorer):
    """Explorer for DyNA-PPO."""

    def __init__(
        self,
        landscape: flexs.Landscape,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        log_file: str = None,
        ensemble_r_squared_threshold: float = 0.5,
        num_experiment_rounds: int = 10,
        num_model_rounds: int = 20,
    ):
        """Explorer which implements DynaPPO.

        Paper: https://openreview.net/pdf?id=HklxbgBKvr

        Attributes:
            threshold: Threshold of R2 scores with which to filter out models in
                the ensemble (default: 0.5).
            num_experiment_rounds: Number of experiment-based policy training rounds.
            num_model_rounds: Number of model-based policy training rounds.
            internal_ensemble: All models currently included in the ensemble.
            internal_ensemble_archs: Corresponding architectures of `internal_ensemble`.
            internal_ensemble_uncertainty: Uncertainty of ensemble predictions.
            internal_ensemble_calls: Number of calls made to the ensemble.
            model_exit_early: Whether or not we should stop model-based training, in
                the event that the model uncertainty is too high.
            ens: All possible models to include in the ensemble.
            ens_archs: Corresponding architectures of `ens`.
            has_learned_policy: Whether or not we have learned a good policy (we need
                to do so before we can propose samples).
            meas_seqs: All measured sequences.
            meas_seqs_it: Iterator through `meas_seqs`.
            top_seqs: Top measured sequences by fitness score.
            top_seqs_it: Iterator through `top_seqs`.
            agent: PPO TF Agent.
            tf_env: Environment in which `agent` operates.
        """

        name = f"DynaPPO_Agent_{ensemble_r_squared_threshold}_{num_experiment_rounds}_{num_model_rounds}"
        model = DynaPPOEnsemble(
            len(starting_sequence), ensemble_r_squared_threshold, alphabet
        )

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
        self.num_experiment_rounds = num_experiment_rounds
        self.num_model_rounds = num_model_rounds

        self.has_learned_policy = False

        self.tf_env = None
        self.agent = None

        env = DynaPPOEnv(
            alphabet=self.alphabet,
            starting_seq=starting_sequence,
            model=model,
            max_num_steps=self.model_queries_per_batch,
        )
        validate_py_environment(env, episodes=1)
        self.tf_env = tf_py_environment.TFPyEnvironment(env)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=[128],
        )
        value_net = value_network.ValueNetwork(
            self.tf_env.observation_spec(), fc_layer_params=[128]
        )

        self.agent = ppo_agent.PPOAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=10,
            summarize_grads_and_vars=False,
        )
        self.agent.initialize()

    def _set_tf_env_reward(self, give_oracle_reward: bool):
        """Set TF-Agents environment reward.

        If `give_oracle_reward` is true, the reward value in the environment is coming from the
        oracle. Otherwise, the reward value comes from the ensemble model.
        """
        self.tf_env.pyenv.envs[0].give_oracle_reward = give_oracle_reward

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
            seq = s_utils.one_hot_to_string(
                experience.observation.numpy()[0], self.alphabet
            )
            new_seqs.add(seq)

            top_fitness = max(new_seqs.values())
            top_sequences = [
                seq for seq, fitness in new_seqs.items() if fitness >= 0.9 * top_fitness
            ]
            self.tf_env.pyenv.envs[0].seq = np.random.choice(top_sequences)

    def perform_model_based_training_step(
        self, measured_sequences_data
    ) -> Union[type(None), bool]:
        """Perform model-based training step."""
        # Change reward to ensemble based.
        self._set_tf_env_reward(give_oracle_reward=False)

        seen_seqs = set(measured_sequences_data["sequence"])
        proposed_seqs = set()

        replay_buffer, collect_driver = self.get_replay_buffer_and_driver(
            new_sequence_bucket=proposed_seqs
        )

        effective_budget = self.sequences_batch_size
        self.ensemble.cost = 0
        while self.ensemble.cost < effective_budget:
            collect_driver.run()
            if self.ensemble.should_model_exit_early:
                return True

        # Train policy on samples collected.
        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()

    def learn_policy(self, measured_sequences_data: pd.DataFrame):
        """Learn policy."""

        for n in range(self.num_experiment_rounds):
            print(f"Experiment based round {n}/{self.num_experiment_rounds}")
            self.perform_experiment_based_training_step(measured_sequences_data)

        self.has_learned_policy = True

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
            ],
            num_episodes=1,
        )

        return replay_buffer, collect_driver

    def propose_sequences(self, measured_sequences_data):
        """Propose `self.sequences_batch_size` samples."""
        if not self.has_learned_policy:
            self.learn_policy(measured_sequences_data)

        # Need to switch back to using the model.
        self._set_tf_env_reward(give_oracle_reward=True)
        seen_seqs = set(measured_sequences_data["sequence"])
        proposed_seqs = set()

        replay_buffer, collect_driver = self.get_replay_buffer_and_driver(
            new_sequence_bucket=proposed_seqs
        )

        # Since we used part of the total budget for pretraining, amortize this cost.
        effective_budget = (
            self.rounds * self.model_queries_per_batch / 2
        ) / self.rounds

        print("Effective budget:", effective_budget)
        previous_model_cost = self.model.cost

        attempts = 0
        cycles = 3
        # Terminate if all we see are old sequences.
        while ((self.model.cost - previous_model_cost) < effective_budget) and (
            attempts < effective_budget * cycles
        ):
            collect_driver.run()
            attempts += 1
        print("Total evals:", self.model.cost - previous_model_cost)

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

        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()

        return (
            measured_proposed_seqs["sequence"].to_numpy()[: self.sequences_batch_size],
            measured_proposed_seqs["model_score"].to_numpy()[
                : self.sequences_batch_size
            ],
        )
