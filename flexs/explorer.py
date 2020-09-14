import abc
import json
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd

import flexs
from typing import Dict, Tuple


class Explorer(abc.ABC):
    def __init__(
        self,
        model: flexs.Model,
        landscape: flexs.Landscape,
        name: str,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        log_file: str = None,
    ):
        self.model = model
        self.landscape = landscape
        self.name = name

        self.rounds = rounds
        self.sequences_batch_size = sequences_batch_size
        self.model_queries_per_batch = model_queries_per_batch
        self.starting_sequence = starting_sequence

        self.log_file = log_file
        if self.log_file is not None:
            self.log_file = Path(self.log_file)
            self.log_file.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def propose_sequences(self, measured_sequences_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Propose a list of sequences to be measured in the next round.

        This method will be overriden to contain the explorer logic for each explorer.

        Args:
            measured_sequences_data: A pandas dataframe of all sequences that have been
            measured by the ground truth so far. Has columns "sequence",
            "true_score", "model_score", and "round".

        Returns:
            (np.ndarray(string), np.ndarray(float)): A tuple containing the proposed
            sequences and their scores (according to the model).
        """
        pass

    def _log(self, metadata: Dict, sequences_data: pd.DataFrame, preds: float, true_score: float, current_round: int, verbose: bool) -> None:
        if self.log_file is not None:
            with open(self.log_file, "w") as f:
                # First write metadata
                json.dump(metadata, f)
                f.write("\n")

                # Then write pandas dataframe
                sequences_data.to_csv(f, index=False)

        if verbose:
            print(f"round: {current_round}, top: {true_score.max()}")

    def run(self, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Run the exporer."""

        self.model.cost = 0

        # Metadata about run that will be used for logging purposes
        metadata = {
            "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
            "exp_name": self.name,
            "model_name": self.model.name,
            "landscape_name": self.landscape.name,
            "rounds": self.rounds,
            "sequences_batch_size": self.sequences_batch_size,
            "model_queries_per_batch": self.model_queries_per_batch,
        }

        # Initial sequences and their scores
        sequences_data = pd.DataFrame(
            {
                "sequence": self.starting_sequence,
                "model_score": np.nan,
                "true_score": self.landscape.get_fitness([self.starting_sequence]),
                "round": 0,
                "model_cost": self.model.cost,
                "measurement_cost": 1,
            }
        )
        self._log(
            sequences_data,
            metadata,
            sequences_data["model_score"],
            sequences_data["true_score"],
            0,
            verbose,
        )

        # For each round, train model on available data, propose sequences,
        # measure them on the true landscape, add to available data, and repeat.
        for r in range(1, self.rounds + 1):
            self.model.train(
                sequences_data["sequence"].to_numpy(), sequences_data["true_score"].to_numpy()
            )

            seqs, preds = self.propose_sequences(sequences_data)
            true_score = self.landscape.get_fitness(seqs)

            if len(seqs) > self.sequences_batch_size:
                raise ValueError(
                    "Must propose <= `self.sequences_batch_size` sequences per round"
                )

            sequences_data = sequences_data.append(
                pd.DataFrame(
                    {
                        "sequence": seqs,
                        "model_score": preds,
                        "true_score": true_score,
                        "round": r,
                        "model_cost": self.model.cost,
                        "measurement_cost": len(sequences_data) + len(seqs),
                    }
                )
            )
            self._log(metadata, sequences_data, preds, true_score, r, verbose)

        return sequences_data, metadata
