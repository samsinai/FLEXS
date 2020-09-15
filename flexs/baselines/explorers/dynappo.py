"""DyNA-PPO explorer."""
import collections
from functools import partial

import numpy as np
import tensorflow as tf
import sklearn
import sklearn.linear_model
import sklearn.gaussian_process
import sklearn.ensemble
import sklearn.tree
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import flexs
from flexs import baselines
from flexs.baselines.explorers.environments.dynappo import (
    DynaPPOEnvironment as DynaPPOEnv,
)
from flexs.utils.sequence_utils import (
    construct_mutant_from_sample,
    one_hot_to_string,
    string_to_one_hot,
)
from flexs.landscape import SEQUENCES_TYPE
import pandas as pd
from typing import List, Set, Tuple, Type, Union


class DynaPPOEnsemble(flexs.Ensemble):
    """Ensemble model and helper for DyNA-PPO."""

    def __init__(
        self,
        ensemble_models: List[flexs.Model],
        explorer_model: flexs.Model,
        explorer_landscape: flexs.Landscape,
        alphabet: str,
        threshold: float,
    ):
        super().__init__(models=ensemble_models)

        self.passing_models = []  # Models which pass the R^2 threshold.

        self.explorer_model = explorer_model
        self.explorer_landscape = explorer_landscape
        self.alphabet = alphabet
        self.threshold = threshold

        self.initial_ensemble_uncertainty = None
        self.should_model_exit_early = False

    def _get_oracle_data(
        self, sequences: SEQUENCES_TYPE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get sequences and fitnesses according to the oracle."""
        return (
            np.array([string_to_one_hot(seq, self.alphabet) for seq in sequences]),
            self.explorer_model.get_fitness(sequences),
        )

    def fit(self, sequences: SEQUENCES_TYPE):
        """Fit internal ensemble to oracle observations.

        This will train all models in the internal ensemble according to the sequences
        and fitnesses observed by the oracle. Then it will filter the models in the
        ensemble according to their R^2 score.
        """
        X, Y = self._get_oracle_data(sequences)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=0
        )

        model_r2s = []
        for model in self.models:
            try:
                model.train(X_train, Y_train)
                Y_pred = model.get_fitness(X_test)
                model_r2s.append(r2_score(Y_test, Y_pred))
            except:  # pylint: disable=bare-except
                # Catch-all for errors; for example, if a KNN fails to fit.
                model_r2s.append(-1)

        self.filter_models(np.array(model_r2s))

    def filter_models(self, model_r2s):
        """Filter models.

        This function filters out models in the ensemble based on whether or not their
        R^2 score passes a predetermined threshold.
        """
        passing = model_r2s > self.threshold
        print(
            f"{np.sum(passing)}/{len(passing)} models passed the {self.threshold} threshold."
        )
        self.passing_models = np.array(self.models)[passing]

    def train(self, sequences: SEQUENCES_TYPE, labels):
        for model in self.models:
            model.train(sequences, labels)

    def _fitness_function(self, sequences: SEQUENCES_TYPE):
        # Because we use the ensemble in the environment to evaluate one sequence at a time, and we need to compute a single number for model uncertainty, we restrict the number of sequences to be one.
        if len(sequences) != 1:
            raise ValueError(
                "Only one sequence should be passed into the ensemble at a time."
            )

        scores = np.stack(
            [model.get_fitness(sequences) for model in self.models], axis=1
        )

        if self.initial_ensemble_uncertainty is None:
            self.initial_ensemble_uncertainty = np.std(reward, axis=1)[0]
        else:
            if np.std(reward, axis=1)[0] > 2 * self.initial_ensemble_uncertainty:
                self.should_model_exit_early = True
        return self.combine_with(scores)


class DynaPPO(flexs.Explorer):
    """Explorer for DyNA-PPO."""

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
        threshold: float = 0.5,
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

        name = f"DynaPPO_Agent_{threshold}_{num_experiment_rounds}_{num_model_rounds}"

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
        self.threshold = threshold
        self.num_experiment_rounds = num_experiment_rounds
        self.num_model_rounds = num_model_rounds

        self.ensemble = None

        self.has_learned_policy = False

        self.tf_env = None
        self.agent = None

    def reset_agent_sequences_data(self, measured_sequence_data: pd.DataFrame):
        self.agent_sequences_data = measured_sequence_data[
            ["sequence", "model_score"]
        ].copy()
        self.agent_sequences_data = self.agent_sequences_data.sort_values(
            by="model_score", ascending=False
        )

    def initialize_env(self):
        """Initialize TF-Agents environment."""
        env = DynaPPOEnv(
            alphabet=self.alphabet,
            starting_seq=self.agent_sequences_data.iloc[0]["sequence"],
            landscape=self.model,
            max_num_steps=self.model_queries_per_batch,
            get_fitness_ensemble=self.ensemble.get_fitness,
            give_oracle_reward=False,
        )

        self.tf_env = tf_py_environment.TFPyEnvironment(env)

    def initialize_ensemble(self):
        """Initialize the internal ensemble.

        DyNA-PPO relies on an ensemble model as a surrogate for the oracle. Here, we
        initialize every supported model architecture and trim them out based on their
        R^2 score.
        """

        seq_len = len(self.starting_sequence)

        ensemble_models = [
            # FLEXS models
            baselines.models.GlobalEpistasisModel(seq_len, 100, self.alphabet),
            baselines.models.MLP(seq_len, 200, self.alphabet),
            baselines.models.CNN(seq_len, 32, 100, self.alphabet),
            # Sklearn models
            baselines.models.LinearRegression(self.alphabet),
            baselines.models.RandomForest(self.alphabet),
            baselines.models.SklearnRegressor(
                sklearn.neighbors.NearestNeighbors(), self.alphabet, "nearest_neighbors"
            ),
            baselines.models.SklearnRegressor(
                sklearn.linear_model.Lasso(), self.alphabet, "lasso"
            ),
            baselines.models.SklearnRegressor(
                sklearn.linear_model.BayesianRidge(), self.alphabet, "bayesian_ridge"
            ),
            baselines.models.SklearnRegressor(
                sklearn.gaussian_process.GaussianProcessRegressor(),
                self.alphabet,
                "gaussian_process",
            ),
            baselines.models.SklearnRegressor(
                sklearn.ensemble.GradientBoostingRegressor(),
                self.alphabet,
                "gradient_boosting",
            ),
            baselines.models.SklearnRegressor(
                sklearn.tree.ExtraTreeRegressor(), self.alphabet, "extra_trees"
            ),
        ]

        self.ensemble = DynaPPOEnsemble(
            ensemble_models=ensemble_models,
            explorer_model=self.model,
            explorer_landscape=self.landscape,
            alphabet=self.alphabet,
            threshold=self.threshold,
        )

    def _set_tf_env_reward(self, give_oracle_reward: bool):
        """Set TF-Agents environment reward.

        If `give_oracle_reward` is true, the reward value in the environment is coming from the
        oracle. Otherwise, the reward value comes from the ensemble model.
        """
        self.tf_env.pyenv.envs[0].give_oracle_reward = give_oracle_reward

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
        self.agent = ppo_agent.PPOAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=num_epochs,
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
            seq = one_hot_to_string(experience.observation.numpy()[0], self.alphabet)
            new_seqs.add(seq)

            self.agent_sequences_data_iter = (self.agent_sequences_data_iter + 1) % len(
                self.agent_sequences_data
            )
            self.tf_env.pyenv.envs[0].seq = self.agent_sequences_data.iloc[
                self.agent_sequences_data_iter
            ]["sequence"]

    def perform_model_based_training_step(self) -> Union[type(None), bool]:
        """Perform model-based training step."""
        # Change reward to ensemble based.
        self._set_tf_env_reward(give_oracle_reward=False)

        seen_seqs = set(self.agent_sequences_data["sequence"])
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

    def perform_experiment_based_training_step(self):
        """Perform experiment-based training step."""
        # Change reward to oracle-based function.
        self._set_tf_env_reward(give_oracle_reward=True)

        seen_seqs = set(self.agent_sequences_data["sequence"])
        proposed_seqs = set()

        replay_buffer, collect_driver = self.get_replay_buffer_and_driver(
            new_sequence_bucket=proposed_seqs
        )

        effective_budget = (self.rounds * self.model_queries_per_batch / 2) / (
            self.num_experiment_rounds
        )

        self.agent_sequences_data_iter = 0

        previous_model_cost = self.model.cost
        iterations = 0
        while (self.model.cost - previous_model_cost) < effective_budget:
            collect_driver.run()
            iterations += 1
            # We've looped over, found nothing new.
            if iterations >= effective_budget and self.agent_sequences_data_iter == 0:
                break

        new_proposed_seqs = list(proposed_seqs.difference(seen_seqs))

        # Train policy on samples collected.
        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()

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

        # Fit candidate models.
        self.ensemble.fit(list(self.agent_sequences_data["sequence"]))
        if len(self.ensemble.passing_models) == 0:
            print("No models passed the threshold. Skipping model-based training.")
            return

        self.ensemble.initial_ensemble_uncertainty = None
        for m in range(self.num_model_rounds):
            print(f"Model based round {m}/{self.num_model_rounds}.")
            should_exit_early = self.perform_model_based_training_step()
            # Early exit because of model uncertainty.
            if should_exit_early:
                print(
                    f"Exiting early at round {m}/{self.num_model_rounds} due to uncertainty."
                )
                break

    def learn_policy(self, measured_sequences_data: pd.DataFrame):
        """Learn policy."""
        self.agent_sequences_data = measured_sequences_data[
            ["sequence", "model_score"]
        ].copy()
        self.agent_sequences_data = self.agent_sequences_data.sort_values(
            by="model_score", ascending=False
        )

        if self.ensemble is None:
            self.initialize_ensemble()
        if self.tf_env is None:
            self.initialize_env()
        if self.agent is None:
            self.initialize_agent()

        for n in range(self.num_experiment_rounds):
            print(f"Experiment based round {n}/{self.num_experiment_rounds}")
            self.perform_experiment_based_training_step()

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

        if len(self.agent_sequences_data) == 0:
            self.reset_agent_sequences_data()

        # Need to switch back to using the model.
        self._set_tf_env_reward(give_oracle_reward=True)
        seen_seqs = set(self.agent_sequences_data["sequence"])
        proposed_seqs = set()

        replay_buffer, collect_driver = self.get_replay_buffer_and_driver(
            new_sequence_bucket=proposed_seqs
        )

        # Reset counter.
        self.agent_sequences_data_iter = 0

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
            # We've looped over, found nothing new.
            if self.agent_sequences_data_iter == 0:
                break
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
