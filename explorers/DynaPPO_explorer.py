import collections
import numpy as np
import tensorflow as tf
from functools import partial
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tf_agents.agents.ppo import ppo_policy, ppo_agent, ppo_utils
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from environments.DynaPPO_environment import DynaPPOEnvironment as DynaPPOEnv
from explorers.base_explorer import Base_explorer
from utils.sequence_utils import (
    translate_one_hot_to_string,
    translate_string_to_one_hot,
)
from utils.model_architectures import (
    Linear,
    NLNN,
    CNNa,
    SKLinear,
    SKLasso,
    SKRF,
    SKGB,
    SKNeighbors,
    SKBR,
    SKGP,
    SKExtraTrees,
)


class DynaPPO_explorer(Base_explorer):
    def __init__(
        self,
        batch_size=100,
        alphabet="UCGA",
        virtual_screen=10,
        threshold=0.5,
        num_experiment_rounds=10,
        num_model_rounds=20,
        path="./simulations/",
        debug=False,
    ):
    """
    DynaPPO implementation.

    The algorithm is:
        for N experiment rounds
            collect samples with policy
            train policy on samples
            fit candidate models on samples and compute R^2
            select models which pass threshold
            if model subset is not empty then
                for M model-based training rounds
                    sample batch of sequences from policy and observe ensemble reward
                    update policy on observed data
    """
        super().__init__(batch_size, alphabet, virtual_screen, path, debug)

        self.explorer_type = (
            f"DynaPPO_Agent_{threshold}_{num_experiment_rounds}_{num_model_rounds}"
        )
        self.threshold = threshold
        self.num_experiment_rounds = num_experiment_rounds
        self.num_model_rounds = num_model_rounds
        self.reset()

    def reset(self):
        self.agent = None
        self.tf_env = None
        self.internal_ensemble = None
        self.internal_ensemble_archs = None

        self.meas_seqs = []
        self.meas_seqs_it = 0

        self.top_seqs = collections.deque(maxlen=self.batch_size)
        self.top_seqs_it = 0

        self.has_learned_policy = False
        self.batches = {-1: ""}

        self.model_uncertainty_low = None
        self.original_model_uncertainty = None
        self.original_horizon = None

    def reset_measured_seqs(self):
        self.meas_seqs = [
            (self.model.get_fitness(seq), seq, self.model.cost)
            for seq in self.model.measured_sequences
        ]
        self.meas_seqs = sorted(self.meas_seqs, key=lambda x: x[0], reverse=True)

    def initialize_env(self):
        env = DynaPPOEnv(
            alphabet=self.alphabet,
            starting_seq=self.meas_seqs[0][1],
            landscape=self.model,
            max_num_steps=self.virtual_screen,
            ensemble_fitness=self.get_internal_ensemble_fitness,
            oracle_reward=False,
        )

        self.tf_env = tf_py_environment.TFPyEnvironment(env)

    def initialize_internal_ensemble(self):
        ens = [
            Linear,
            CNNa,
            SKLinear,
            SKLasso,
            SKRF,
            SKGB,
            SKNeighbors,
            SKBR,
            SKGP,
            SKExtraTrees,
        ]
        ens_archs = [
            Linear,
            CNNa,
            SKLinear,
            SKLasso,
            SKRF,
            SKGB,
            SKNeighbors,
            SKBR,
            SKGP,
            SKExtraTrees,
        ]

        ens_archs = [
            arch(len(self.meas_seqs[0][1]), alphabet=self.alphabet)
            for arch in ens_archs
        ]
        ens = [arch.get_model() for arch in ens_archs]
        self.ens = ens
        self.ens_archs = ens_archs

        self.internal_ensemble = ens
        self.internal_ensemble_archs = ens_archs
        self.internal_ensemble_calls = 0
        self.internal_ensemble_uncertainty = None
        self.model_exit_early = None

    def set_tf_env_reward(self, is_oracle):
        self.tf_env.pyenv.envs[0].oracle_reward = is_oracle

    def initialize_agent(self):
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
        """
        Given a trajectory object, checks if
        the object is the last in the trajectory,
        then adds the sequence corresponding
        to the state to batch.

        If the episode is ending, it changes the
        "current sequence" of the environment
        to the next one in `last_batch`,
        so that when the environment resets, mutants
        are generated from that new sequence.
        """

        if experience.is_boundary():
            seq = translate_one_hot_to_string(
                experience.observation.numpy()[0], self.alphabet
            )
            new_seqs.add(seq)

            self.meas_seqs_it = (self.meas_seqs_it + 1) % len(self.meas_seqs)
            self.tf_env.pyenv.envs[0].seq = self.meas_seqs[self.meas_seqs_it][1]

    def get_oracle_sequences_and_fitness(self, sequences):
        X = []
        Y = []

        for sequence in sequences:
            x = translate_string_to_one_hot(sequence, self.alphabet)
            y = self.model.get_fitness(sequence)

            X.append(x)
            Y.append(y)

        return np.array(X), np.array(Y)

    def get_internal_ensemble_fitness(self, sequence):
        reward = []

        for i, (model, arch) in enumerate(
            zip(self.internal_ensemble, self.internal_ensemble_archs)
        ):
            x = np.array(translate_string_to_one_hot(sequence, self.alphabet))
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
        X, Y = self.get_oracle_sequences_and_fitness(sequences)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=0
        )

        internal_r2s = []

        for i, (model, arch) in enumerate(zip(self.ens, self.ens_archs)):
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
            except:
                # for example KNN failed to fit
                internal_r2s.append(-1)

        self.filter_models(np.array(internal_r2s))

    def filter_models(self, r2s):
        print(r2s)
        # filter out models by threshold
        good = r2s > self.threshold
        print(f"Allowing {np.sum(good)}/{len(good)} models.")
        self.internal_ensemble_archs = np.array(self.ens_archs)[good]
        self.internal_ensemble = np.array(self.ens)[good]

    def perform_model_based_training_step(self):
        # change reward to ensemble based
        self.set_tf_env_reward(is_oracle=False)

        all_seqs = set(self.model.measured_sequences)
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

        effective_budget = self.batch_size
        self.internal_ensemble_calls = 0
        while self.internal_ensemble_calls < effective_budget:
            collect_driver.run()
            if self.model_exit_early:
                return -1

        new_seqs = new_seqs.difference(all_seqs)

        # train policy on samples collected
        trajectories = replay_buffer.gather_all()
        total_loss, _ = self.agent.train(experience=trajectories)
        replay_buffer.clear()

        return 1

    def perform_experiment_based_training_step(self):
        # change reward to oracle based
        self.set_tf_env_reward(is_oracle=True)

        all_seqs = set(self.model.measured_sequences)
        new_seqs = set()
        last_batch = self.get_last_batch()

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
            self.original_horizon * self.batch_size * self.virtual_screen / 2
        ) / (self.num_experiment_rounds)

        self.meas_seqs_it = 0

        previous_evals = self.model.evals
        iterations = 0
        while (self.model.evals - previous_evals) < effective_budget:
            collect_driver.run()
            iterations += 1
            # we've looped over, found nothing new
            if iterations >= effective_budget and self.meas_seqs_it == 0:
                break

        new_seqs = new_seqs.difference(all_seqs)

        # train policy on samples collected
        trajectories = replay_buffer.gather_all()
        total_loss, _ = self.agent.train(experience=trajectories)
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
            v = self.perform_model_based_training_step()
            # early exit because of model uncertainty
            if v == -1:
                print(
                    f"Exiting early at round {m}/{self.num_model_rounds} due to uncertainty."
                )
                break

    def learn_policy(self):
        self.meas_seqs = [
            (self.model.get_fitness(seq), seq, self.model.cost)
            for seq in self.model.measured_sequences
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
            self.perform_experiment_based_training_step()

        self.has_learned_policy = True

    def propose_samples(self):
        if self.original_horizon is None:
            self.original_horizon = self.horizon

        if not self.has_learned_policy:
            self.learn_policy()

        if len(self.meas_seqs) == 0:
            self.reset_measured_seqs()

        # need to switch back to using the model
        self.set_tf_env_reward(is_oracle=True)
        all_seqs = set(self.model.measured_sequences)
        new_seqs = set()
        last_batch = self.get_last_batch()

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
            self.original_horizon * self.batch_size * self.virtual_screen / 2
        ) / self.original_horizon

        print("Effective budget:", effective_budget)
        previous_evals = self.model.evals
        while (self.model.evals - previous_evals) < effective_budget:
            collect_driver.run()
            # we've looped over, found nothing new
            if self.meas_seqs_it == 0:
                break
        print("Total evals:", self.model.evals - previous_evals)

        new_seqs = new_seqs.difference(all_seqs)

        # add new sequences to measured_sequences and sort
        new_meas_seqs = [
            (self.model.get_fitness(seq), seq, self.model.cost) for seq in new_seqs
        ]
        new_meas_seqs = sorted(new_meas_seqs, key=lambda x: x[0], reverse=True)
        self.meas_seqs += new_meas_seqs
        self.meas_seqs = sorted(self.meas_seqs, key=lambda x: x[0], reverse=True)

        trajectories = replay_buffer.gather_all()
        total_loss, _ = self.agent.train(experience=trajectories)
        replay_buffer.clear()

        return [s[1] for s in new_meas_seqs[: self.batch_size]]
