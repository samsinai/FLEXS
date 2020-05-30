"""Random explorer."""
from explorers.base_explorer import Base_explorer
from utils.sequence_utils import generate_random_mutant


class Random_explorer(Base_explorer):
    """Random explorer.

    Parameters:
        mu (float): The probability that each position gets mutated.
    """

    def __init__(
        self,
        mu,
        batch_size=100,
        alphabet="UCGA",
        virtual_screen=10,
        path="./simulations/",
        debug=False,
    ):
        """Initialize."""
        super(Random_explorer, self).__init__(
            batch_size, alphabet, virtual_screen, path, debug
        )
        self.mu = mu
        self.explorer_type = f"Random_mu{self.mu}"

    def propose_samples(self):
        """Propose."""
        new_seqs = set()
        last_batch = self.get_last_batch()
        while len(new_seqs) < self.batch_size:
            for seq in self.batches[last_batch]:
                new_seq = generate_random_mutant(
                    seq, self.mu * 1.0 / len(seq), alphabet=self.alphabet
                )
                if new_seq not in self.model.measured_sequences:
                    new_seqs.add(new_seq)
        return new_seqs
