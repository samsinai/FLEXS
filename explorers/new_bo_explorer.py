import numpy as np

from explorers.base_explorer import Base_explorer
from utils.sequence_utils import translate_string_to_one_hot

# new BO stuff


class New_BO_Explorer(Base_explorer):
    """
    Bayesian optimization explorer. Uses Gaussian process with Matern kernel
    on black box function.

    Reference: http://krasserm.github.io/2018/03/21/bayesian-optimization/
    """

    def __init__(
        self,
        batch_size=100,
        alphabet="UCGA",
        virtual_screen=10,
        path="./simulations/",
        debug=False,
        method="EI",
    ):
        super(New_BO_Explorer, self).__init__(
            batch_size=batch_size,
            alphabet=alphabet,
            virtual_screen=virtual_screen,
            path=path,
            debug=debug,
        )
        self.explorer_type = "New_BO_Explorer"
        self.alphabet_len = len(alphabet)
        self.method = method
        self.best_fitness = 0
        self.top_sequence = []

        self.seq_len = None
        self.maxima = None
        self._reset = False

    def _initialize(self):
        start_sequence = list(self.model.measured_sequences)[0]
        #     self.state = translate_string_to_one_hot(start_sequence, self.alphabet)
        self.seq_len = len(start_sequence)

    def reset(self):
        self.best_fitness = 0
        self.batches = {-1: ""}
        self._reset = True

    def propose_sequences_via_thompson(self):
        """
        Proposes a batch of new sequences based on Thompson sampling with
        a Gaussian posterior.
        """

        print("Enumerating all sequences in the space.")

        self.maxima = []

        def enum_and_eval(curr_seq):
            # if we have a full sequence, then let's evaluate
            if len(curr_seq) == self.seq_len:
                mu, sigma = self.model.get_fitness(curr_seq, return_std=True)
                estimated_fitness = np.random.normal(mu, sigma)
                self.maxima.append([estimated_fitness, curr_seq])
            else:
                for char in list(self.alphabet):
                    enum_and_eval(curr_seq + char)

        enum_and_eval("")

        # Sort descending based on the value.
        return sorted(self.maxima, reverse=True, key=lambda x: x[0])

    def propose_sequences_via_greedy(self):
        """
        Proposes a batch of new sequences based on greedy in the expectation of the
        Gaussian posterior.
        """

        print("Enumerating all sequences in the space.")

        self.maxima = []

        def enum_and_eval(curr_seq):
            # if we have a full sequence, then let's evaluate
            if len(curr_seq) == self.seq_len:
                mu = self.model.get_fitness(curr_seq, return_std=False)
                self.maxima.append([mu, curr_seq])
            else:
                for char in list(self.alphabet):
                    enum_and_eval(curr_seq + char)

        enum_and_eval("")

        # Sort descending based on the value.
        return sorted(self.maxima, reverse=True, key=lambda x: x[0])

    def propose_sequences_via_ucb(self):
        """
        Proposes a batch of new sequences based on greedy in the expectation of the
        Gaussian posterior.
        """

        print("Enumerating all sequences in the space.")

        self.maxima = []

        def enum_and_eval(curr_seq):
            # if we have a full sequence, then let's evaluate
            if len(curr_seq) == self.seq_len:
                mu, sigma = self.model.get_fitness(curr_seq, return_std=True)
                self.maxima.append([mu + 0.01 * sigma, curr_seq])
            else:
                for char in list(self.alphabet):
                    enum_and_eval(curr_seq + char)

        enum_and_eval("")

        # Sort descending based on the value.
        return sorted(self.maxima, reverse=True, key=lambda x: x[0])

    def propose_samples(self):
        if self._reset:
            # indicates model was reset
            self._initialize()
        self._reset = False

        samples = set()

        new_seqs = self.propose_sequences_via_thompson()
        new_states = []
        new_fitnesses = []
        i = 0
        while (len(new_states) < self.batch_size) and i < len(new_seqs):
            new_fitness, new_seq = new_seqs[i]
            if new_seq not in self.model.measured_sequences:
                new_state = translate_string_to_one_hot(new_seq, self.alphabet)
                if new_fitness >= self.best_fitness:
                    self.top_sequence.append((new_fitness, new_state, self.model.cost))
                    self.best_fitness = new_fitness
                samples.add(new_seq)
                new_states.append(new_state)
                new_fitnesses.append(new_fitness)
            i += 1

        print("Current best fitness:", self.best_fitness)

        return list(samples)
