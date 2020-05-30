"""Base explorer."""

import os.path
import uuid

from meta.explorer import Explorer


class Base_explorer(Explorer):
    """Base explorer."""
    def __init__(
        self,
        batch_size=100,
        alphabet="UCGA",
        virtual_screen=10,
        path="./simulations/",
        debug=False,
    ):
        """Initialize."""
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.virtual_screen = virtual_screen
        self.batches = {-1: ""}
        self.explorer_type = "Base"
        self.horizon = 1
        self.path = path
        self.debug = debug
        self.run_id = str(uuid.uuid1())
        self.model = None

    @property
    def file_to_write(self):
        """File."""
        return f"{self.path}{self.explorer_type}.csv"

    def write(self, current_round, overwrite):
        """Write."""
        if not os.path.exists(self.file_to_write) or (current_round == 0 and overwrite):
            with open(self.file_to_write, "w") as output_file:
                output_file.write(
                    "id,batch,sequence,true_score,model_score,batch_size,"
                    "measurement_cost,virtual_evals,landscape_id,start_id,"
                    "model_type,virtual_screen,horizon,explorer_type\n"
                )

        with open(self.file_to_write, "a") as output_file:
            batch = self.get_last_batch()
            for sequence in self.batches[batch]:
                output_file.write(
                    f"{self.run_id},{batch},{sequence},"
                    f"{self.batches[batch][sequence][1]},"
                    f"{self.batches[batch][sequence][0]},{self.batch_size},"
                    f"{self.model.cost},{self.model.evals},{self.model.landscape_id},"
                    f"{self.model.start_id},{self.model.model_type},"
                    f"{self.virtual_screen}, {self.horizon},{self.explorer_type}\n"
                )

    def get_last_batch(self):
        """Get."""
        return max(self.batches.keys())

    def set_model(self, model, reset=True):  # pylint: disable=W0221
        """Set."""
        if reset:
            self.reset()
        self.model = model
        if self.model.cost > 0:
            batch = self.get_last_batch() + 1
            self.batches[batch] = {}
            for seq in self.model.measured_sequences:
                score = self.model.get_fitness(seq)
                self.batches[batch][seq] = [score, score]

    def propose_samples(self):
        """Implement this function for your own explorer."""
        raise NotImplementedError(
            "propose_samples must be implemented by your explorer"
        )

    def measure_proposals(self, proposals):
        """Measure."""
        to_measure = list(proposals)[: self.batch_size]
        last_batch = self.get_last_batch()
        self.batches[last_batch + 1] = {}
        for seq in to_measure:
            self.batches[last_batch + 1][seq] = [self.model.get_fitness(seq)]
        self.model.update_model(to_measure)
        for seq in to_measure:
            self.batches[last_batch + 1][seq].append(self.model.get_fitness(seq))

    def reset(self):
        """Reset the explorer.

        Overwrite if you are working with a stateful explorer.
        """
        self.batches = {-1: ""}

    def run(self, rounds, overwrite=False, verbose=True):
        """Run."""
        self.horizon = rounds
        if not self.debug:
            self.write(0, overwrite)
        for r in range(rounds):
            if verbose:
                print(
                    f"round: {r}, cost: {self.model.cost}, evals: {self.model.evals}, "
                    f"top: {max(self.model.measured_sequences.values())}"
                )
            new_samples = self.propose_samples()
            self.measure_proposals(new_samples)
            if not self.debug:
                self.write(r, overwrite)
            self.horizon -= 1
