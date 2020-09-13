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
from flexs.baselines.explorers.environments.DynaPPO_environment import (
    DynaPPOEnvironment as DynaPPOEnv,
)
from flexs.utils import sequence_utils as s_utils


class DynaPPO(flexs.Explorer):
    """Explorer for DyNA-PPO."""

    def __init__(
        self,
        model,
        landscape,
        rounds,
        sequences_batch_size,
        model_queries_per_batch,
        starting_sequence,
        alphabet,
        log_file=None,
        threshold=0.5,
        num_experiment_rounds=10,
        num_model_rounds=20,
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

        self.model_exit_early = None

        self.internal_ensemble = None
        self.internal_ensemble_archs = None
        self.internal_ensemble_uncertainty = None
        self.internal_ensemble_calls = None

        self.ens = None
        self.original_horizon = self.rounds
        self.has_learned_policy = None

        self.meas_seqs = []
        self.meas_seqs_it = 0
        self.top_seqs = collections.deque(maxlen=self.sequences_batch_size)
        self.top_seqs_it = 0

        self.agent = None
        self.tf_env = None

    def initialize_env(self):
        """Initialize TF-Agents environment."""
        env = DynaPPOEnv(
            alphabet=self.alphabet,
            starting_seq=self.meas_seqs[0][1],
            landscape=self.model,
            max_num_steps=self.model_queries_per_batch,
            ensemble_fitness=self.get_internal_ensemble_fitness,
            oracle_reward=False,
        )

        self.tf_env = tf_py_environment.TFPyEnvironment(env)

    def initialize_internal_ensemble(self):
        """Initialize the internal ensemble.

        DyNA-PPO relies on an ensemble model as a surrogate for the oracle. Here, we
        initialize every supported model architecture and trim them out based on their
        R^2 score.
        """

        seq_len = len(self.starting_sequence)

        ens = [
            baselines.models.LinearRegression(self.alphabet),
            baselines.models.RandomForest(self.alphabet),
            baselines.models.GlobalEpistasisModel(seq_len, 100, self.alphabet),
            baselines.models.MLP(seq_len, 200, self.alphabet),
            baselines.models.CNN(seq_len, 32, 100, self.alphabet),
            # Custom sklearn models
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

        self.ens = ens

        self.internal_ensemble = ens
        self.internal_ensemble_calls = 0
        self.internal_ensemble_uncertainty = None

    def set_tf_env_reward(self, is_oracle):
        """Set TF-Agents environment reward.

        If `is_oracle` is true, the reward value in the environment is coming from the
        oracle. Otherwise, the reward value comes from the ensemble model.
        """
        self.tf_env.pyenv.envs[0].oracle_reward = is_oracle

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
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5),
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
            seq = s_utils.one_hot_to_string(
                experience.observation.numpy()[0], self.alphabet
            )
            new_seqs.add(seq)

            self.meas_seqs_it = (self.meas_seqs_it + 1) % len(self.meas_seqs)
            self.tf_env.pyenv.envs[0].seq = self.meas_seqs[self.meas_seqs_it][1]

    def get_oracle_sequences_and_fitness(self, sequences):
        """Get sequences and fitnesses according to the oracle."""
        X = []
        Y = []

        for sequence in sequences:
            x = s_utils.string_to_one_hot(sequence, self.alphabet)
            y = self.model.get_fitness(sequence)

            X.append(x)
            Y.append(y)

        return np.array(X), np.array(Y)

    def get_internal_ensemble_fitness(self, sequence):
        """Get the fitness of a sequence according to the internal ensemble model.

        If the uncertainty of the reward returned by the ensemble model is too high,
        then we stop model-based training.
        """
        reward = []

        for (model, arch) in zip(self.internal_ensemble, self.internal_ensemble_archs):
            x = np.array(s_utils.string_to_one_hot(sequence, self.alphabet))
            if type(arch) in [Linear, NLNN, CNNa]:
                reward.append(max(min(200, model.predict(x[np.newaxis])[0]), -200))
            elif type(arch) in [
                SKLinear,
                SKLasso,
                SKRF,
                SKGB,
                SKNeighbors,
                SKBR,
                SKGP,
                SKExtraTrees,
            ]:
                x = x.reshape(1, np.prod(x.shape))
                reward.append(max(min(200, model.predict(x)[0]), -200))
            else:
                raise ValueError(type(arch))
        self.internal_ensemble_calls += 1
        if self.internal_ensemble_uncertainty is None:
            self.internal_ensemble_uncertainty = np.std(reward)
        else:
            if np.std(reward) > 2 * self.internal_ensemble_uncertainty:
                self.model_exit_early = True
        return np.sum(reward) / len(reward)

    def fit_internal_ensemble(self, sequences):
        """Fit internal ensemble to oracle observations.

        This will train all models in the internal ensemble according to the sequences
        and fitnesses observed by the oracle. Then it will filter the models in the
        ensemble according to their R^2 score.
        """
        X, Y = self.get_oracle_sequences_and_fitness(sequences)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=0
        )

        internal_r2s = []

        for (model, arch) in zip(self.ens, self.ens_archs):
            try:
                if type(arch) in [Linear, NLNN, CNNa]:
                    model.fit(
                        X_train,
                        Y_train,
                        epochs=arch.epochs,
                        validation_split=arch.validation_split,
                        batch_size=arch.batch_size,
                        verbose=0,
                    )
                    y_pred = model.predict(X_test)
                    internal_r2s.append(r2_score(Y_test, y_pred))
                elif type(arch) in [
                    SKLinear,
                    SKLasso,
                    SKRF,
                    SKGB,
                    SKNeighbors,
                    SKBR,
                    SKExtraTrees,
                    SKGP,
                ]:
                    _X = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
                    _X_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))
                    model.fit(_X, Y_train)
                    y_pred = model.predict(_X_test)
                    internal_r2s.append(r2_score(Y_test, y_pred))
                else:
                    raise ValueError(type(arch))
            except:  # pylint: disable=bare-except
                # Catch-all for errors; for example, if a KNN fails to fit.
                internal_r2s.append(-1)

        self.filter_models(np.array(internal_r2s))

    def filter_models(self, r2s):
        """Filter models.

        This function filters out models in the ensemble based on whether or not their
        R^2 score passes a predetermined threshold.
        """
        good = r2s > self.threshold
        print(f"Allowing {np.sum(good)}/{len(good)} models.")
        self.internal_ensemble_archs = np.array(self.ens_archs)[good]
        self.internal_ensemble = np.array(self.ens)[good]

    def perform_model_based_training_step(self, measured_sequences):
        """Perform model-based training step."""
        # Change reward to ensemble based.
        self.set_tf_env_reward(is_oracle=False)

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

        effective_budget = self.sequences_batch_size
        self.internal_ensemble_calls = 0
        while self.internal_ensemble_calls < effective_budget:
            collect_driver.run()
            if self.model_exit_early:
                return -1

        new_seqs = new_seqs.difference(all_seqs)

        # Train policy on samples collected.
        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()

        return 1

    def perform_experiment_based_training_step(self, measured_sequences):
        """Perform experiment-based training step."""
        # Change reward to oracle-based function.
        self.set_tf_env_reward(is_oracle=True)

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

        effective_budget = (
            self.original_horizon * self.model_queries_per_batch / 2
        ) / (self.num_experiment_rounds)

        self.meas_seqs_it = 0

        previous_evals = self.model.cost
        iterations = 0
        while (self.model.cost - previous_evals) < effective_budget:
            collect_driver.run()
            iterations += 1
            # We've looped over, found nothing new.
            if iterations >= effective_budget and self.meas_seqs_it == 0:
                break

        new_seqs = new_seqs.difference(all_seqs)

        # Train policy on samples collected.
        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()

        self.meas_seqs += [
            (self.model.get_fitness(seq), seq, self.model.cost) for seq in new_seqs
        ]
        self.meas_seqs = sorted(self.meas_seqs, key=lambda x: x[0], reverse=True)

        # fit candidate models
        self.fit_internal_ensemble([s[1] for s in self.meas_seqs])
        if len(self.internal_ensemble) == 0:
            print("No ensembles passed. Skipping model-based training.")
            return

        self.internal_ensemble_uncertainty = None
        for m in range(self.num_model_rounds):
            print(f"Model based round {m}/{self.num_model_rounds}.")
            v = self.perform_model_based_training_step(measured_sequences)
            # Early exit because of model uncertainty.
            if v == -1:
                print(
                    f"Exiting early at round {m}/{self.num_model_rounds} due to uncertainty."
                )
                break

    def learn_policy(self, measured_sequences):
        """Learn policy."""
        self.meas_seqs = [
            (self.model.get_fitness(seq), seq, self.model.cost)
            for seq in measured_sequences["sequence"]
        ]
        self.meas_seqs = sorted(self.meas_seqs, key=lambda x: x[0], reverse=True)

        if self.tf_env is None:
            self.initialize_env()
        if self.agent is None:
            self.initialize_agent()
        if self.internal_ensemble is None:
            self.initialize_internal_ensemble()

        for n in range(self.num_experiment_rounds):
            print(f"Experiment based round {n}/{self.num_experiment_rounds}")
            self.perform_experiment_based_training_step(measured_sequences)

        self.has_learned_policy = True

    def propose_sequences(self, measured_sequences):
        """Propose `self.sequences_batch_size` samples."""
        if self.original_horizon is None:
            self.original_horizon = self.rounds

        if not self.has_learned_policy:
            self.learn_policy(measured_sequences)

        if len(self.meas_seqs) == 0:
            meas_seqs = zip(
                measured_sequences["sequence"], measured_sequences["true_score"]
            )
            self.meas_seqs = sorted(meas_seqs, key=lambda x: x[0], reverse=True)

        # Need to switch back to using the model.
        self.set_tf_env_reward(is_oracle=True)
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

        # Reset counter.
        self.meas_seqs_it = 0

        # Since we used part of the total budget for pretraining, amortize this cost.
        effective_budget = (
            self.original_horizon * self.model_queries_per_batch / 2
        ) / self.original_horizon

        print("Effective budget:", effective_budget)
        previous_evals = self.model.cost

        attempts = 0

        # Terminate if all we see are old sequences.
        while ((self.model.cost - previous_evals) < effective_budget) and (
            attempts < effective_budget * 3
        ):
            collect_driver.run()
            attempts += 1
            # We've looped over, found nothing new.
            if self.meas_seqs_it == 0:
                break
        print("Total evals:", self.model.cost - previous_evals)

        new_seqs = new_seqs.difference(all_seqs)

        # If we only saw old sequences, randomly generate to fill the batch.
        while len(new_seqs) < self.sequences_batch_size:
            seqs = s_utils.generate_random_sequences(
                self.seq_len,
                self.sequences_batch_size - len(new_seqs),
                alphabet=self.alphabet,
            )
            for seq in seqs:
                if (seq in new_seqs) or (seq in all_seqs):
                    continue
                new_seqs.add(seq)

        # Add new sequences to `measured_sequences` and sort.
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
