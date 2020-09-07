import abc
import json
from datetime import datetime

import numpy as np
import pandas as pd


class Explorer(abc.ABC):
    def __init__(
        self,
        model,
        landscape,
        name,
        rounds,
        experiment_budget,
        query_budget,
        initial_sequence_data,
        log_file=None,
    ):
        self.model = model
        self.landscape = landscape
        self.name = name

        self.rounds = rounds
        self.experiment_budget = experiment_budget
        self.query_budget = query_budget
        self.initial_sequence_data = initial_sequence_data
        self.log_file = log_file

    @abc.abstractmethod
    def propose_sequences(self, measured_sequences):
        pass

    def _log(self, metadata, sequences, preds, true_score, current_round, verbose):
        if self.log_file is not None:
            with open(self.log_file, "w") as f:
                json.dumps(metadata, f)
                f.write("\n")
                sequences.to_csv(f, index=False)

        if verbose:
            print(f"round: {current_round}, top: {true_score.max()}")

    def run(self, verbose=True):
        """Run the exporer."""

        self.model.cost = 0

        # Metadata about run that will be used for logging purposes
        metadata = {
            "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
            "exp_name": self.name,
            "model_name": self.model.name,
            "landscape_name": self.landscape.name,
            "rounds": self.rounds,
            "experiment_budget": self.experiment_budget,
            "query_budget": self.query_budget,
        }

        # Initial sequences and their scores
        sequences = pd.DataFrame(
            {
                "sequence": self.initial_sequence_data,
                "model_score": np.nan,
                "true_score": self.landscape.get_fitness(self.initial_sequence_data),
                "round": 0,
                "model_cost": self.model.cost,
                "measurement_cost": len(self.initial_sequence_data),
                "landscape": self.landscape.name,
                "model": self.model.name,
            }
        )
        self._log(
            metadata,
            sequences,
            sequences["model_score"],
            sequences["true_score"],
            0,
            verbose,
        )

        # For each round, train model on available data, propose sequences,
        # measure them on the true landscape, add to available data, and repeat.
        for r in range(1, self.rounds + 1):
            self.model.train(
                sequences["sequence"].to_numpy(), sequences["true_score"].to_numpy()
            )

            seqs, preds = self.propose_sequences(sequences)
            true_score = self.landscape.get_fitness(seqs)

            if len(seqs) > self.experiment_budget:
                raise ValueError(
                    "Must propose <= `self.experiment_budget` sequences per round"
                )

            sequences = sequences.append(
                pd.DataFrame(
                    {
                        "sequence": seqs,
                        "model_score": preds,
                        "true_score": true_score,
                        "round": r,
                        "model_cost": self.model.cost,
                        "measurement_cost": len(sequences) + len(seqs),
                        "landscape": self.landscape.name,
                        "model": self.model.name,
                    }
                )
            )
            self._log(metadata, sequences, preds, true_score, r, verbose)

        return sequences, metadata
