"""CMAES explorer."""
import numpy as np

import flexs
import flexs.utils.sequence_utils as s_utils


class CMAES(flexs.Explorer):
    """An explorer which implements the covariance matrix adaptation evolution strategy.

    http://blog.otoro.net/2017/10/29/visual-evolution-strategies/ is a helpful guide
    """

    def __init__(
        self,
        model,
        landscape,
        rounds,
        sequences_batch_size,
        model_queries_per_batch,
        starting_sequence,
        alphabet,
        population_size,
        elite_proportion,
        log_file=None,
    ):
        name = f"CMAES"

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

        self.seq_len = len(self.starting_sequence)
        self.alphabet = alphabet
        self.N = None
        self.mu = None
        self.weights = None
        self.mueff = None
        self.c1 = None
        self.cc = None
        self.cs = None
        self.cmu = None
        self.damp = None
        self.mean = None
        self.sigma = None
        self.cov = None
        self.ps = None
        self.pc = None
        self.chiN = None
        self.old_mean = None

        self.population_size = population_size
        self.elite_proportion = elite_proportion
        self.round = 0

    def initialize_params(self):
        """Initialize all parameters."""
        # to be called after set_model

        self.lam = self.sequences_batch_size

        # we'll be working with one-hots
        N = self.seq_len * len(self.alphabet)
        self.N = N
        self.mu = self.lam // 2
        self.weights = np.array(
            [np.log(self.mu + 0.5) - np.log(i) for i in range(1, self.mu + 1)]
        )
        self.weights /= sum(self.weights)
        self.mueff = sum(self.weights) ** 2 / sum([w ** 2 for w in self.weights])

        self.c1 = 2 / ((self.N + 1.3) ** 2 + self.mueff)
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + self.mueff),
        )
        self.damp = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (N + 1)) - 1) + self.cs

        self.mean = np.zeros(N)
        self.sigma = 0.3
        self.cov = np.identity(N)
        self.ps = np.zeros(N)
        self.pc = np.zeros(N)

        self.chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

    def convert_mvn_to_seq(self, mvn):
        """Convert a multivariate normal sample to a one-hot representation."""
        mvn = mvn.reshape((len(self.alphabet), self.seq_len))
        one_hot = np.zeros((len(self.alphabet), self.seq_len))
        amax = np.argmax(mvn, axis=0)

        for i in range(self.seq_len):
            one_hot[amax[i]][i] = 1

        return s_utils.translate_one_hot_to_string(one_hot, self.alphabet)

    def _sample(self, previous_sequences):
        sequences = {}

        new_sequences = 0
        attempts = 0

        # Terminate if all we see are old sequences.
        while (
            len(sequences) < self.population_size
            and attempts < self.population_size * 3
        ):
            attempts += 1
            x = np.random.multivariate_normal(self.mean, (self.sigma ** 2) * self.cov)
            seq = self.convert_mvn_to_seq(x)

            if seq in previous_sequences or seq in sequences:
                continue

            fitness = self.model.get_fitness([seq]).item()
            new_sequences += 1

            sequences[seq] = fitness

        # If we only saw old sequences, randomly generate to fill the batch.
        while len(sequences) < self.population_size:
            seqs = s_utils.generate_random_sequences(
                self.seq_len,
                self.ground_truth_measurements_per_round - len(sequences),
                alphabet=self.alphabet,
            )
            for seq in seqs:
                if seq in previous_sequences or seq in sequences:
                    continue
                fitness = self.model.get_fitness([seq]).item()
                sequences[seq] = fitness

        return list(sequences.items())

    def compute_new_mean(self, samples):
        """Helper function to recompute mean."""
        s = np.zeros(self.mean.shape)

        for i in range(self.mu):
            s += self.weights[i - 1] * samples[i][1]

        # THIS NORMALIZATION IS NON-STANDARD
        self.mean = s / np.linalg.norm(s)

    def expectation(self):
        """Helper function to approximate expectation."""
        return np.sqrt(self.N) * (1 - 1 / (4 * self.N) + 1 / (21 * self.N ** 2))

    def update_isotropic_evolution_path(self):
        """Helper function to update isotropic evolution path."""
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * np.linalg.inv(np.sqrt(self.cov)) * (self.mean - self.old_mean) / self.sigma

    def update_step_size(self):
        """Update the step size."""
        self.sigma = self.sigma * np.exp(
            (self.cs / self.damp) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )

    def ps_indicator(self):
        """Return indicator of ps."""
        return int(
            np.linalg.norm(self.ps)
            / np.sqrt(1 - (1 - self.cs) ** (2 * self.round / self.lam))
            / self.chiN
            < 1.4 + 2 / (self.N + 1)
        )

    def update_anisotropic_evolution_path(self):
        """Helper function to update anisotropic evolution path."""
        self.pc = (1 - self.cc) * self.pc + self.ps_indicator() * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * (self.mean - self.old_mean) / self.sigma

    def update_covariance_matrix(self, samples):
        """Update the covariance matrix."""
        weighted_sum = sum(
            [
                self.weights[i - 1]
                * ((samples[i][0] - self.old_mean) / self.sigma)
                * ((samples[i][0] - self.old_mean) / self.sigma).T
                for i in range(self.mu)
            ]
        )

        self.cov = (
            (1 - self.c1 - self.cmu) * self.cov
            + self.c1
            * (
                self.pc * self.pc.T
                + (1 - self.ps_indicator()) * self.cc * (2 - self.cc) * self.cov
            )
            + self.cmu * weighted_sum
        )

    def propose_sequences(self, measured_sequences):
        # in CMAES we _minimize_ an objective, so we'll conveniently reverse

        measured_sequence_dict = dict(
            zip(measured_sequences["sequence"], measured_sequences["true_score"])
        )
        self.initialize_params()

        sequences = {}
        for _ in range(int(self.model_queries_per_round / self.population_size)):
            samples = sorted(
                self._sample({**measured_sequence_dict, **sequences}),
                key=lambda s: s[1],
                reverse=True,
            )
            sequences.update(samples)

            elite_size = int(self.elite_proportion * self.population_size)

            self.old_mean = self.mean
            self.compute_new_mean(samples[:elite_size])
            self.update_isotropic_evolution_path()
            self.update_anisotropic_evolution_path()
            self.update_covariance_matrix(samples[:elite_size])
            self.update_step_size()

        # We propose the top `self.ground_truth_measurements_per_round` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[
            : -self.ground_truth_measurements_per_round : -1
        ]
        self.round += 1

        return new_seqs[sorted_order], preds[sorted_order]
