import abc

import numpy as np
import pandas as pd
from datetime import datetime


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
    ):
        self.model = model
        self.landscape = landscape
        self.name = name

        self.rounds = rounds
        self.experiment_budget = experiment_budget
        self.query_budget = query_budget
        self.initial_sequence_data = initial_sequence_data
        self.run_id = datetime.now().strftime("%H:%M:%S-%m/%d/%Y")

    @abc.abstractmethod
    def propose_sequences(self, batches):
        pass

    def run(self, verbose=True):
        """Run the exporer."""

        metadata = {"run_id": self.run_id}
        sequences = pd.DataFrame(
            {
                "sequence": self.initial_sequence_data,
                "model_score": np.nan,
                "true_score": self.landscape.get_fitness(self.initial_sequence_data),
                "round": 0,
                "model_cost": self.model.cost,
                "measurement_cost": len(self.initial_sequence_data),
            }
        )

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
                    }
                )
            )

            if verbose:
                print(f"round: {r}, top: {true_score.max()}")

        return sequences, metadata
