import abc
import uuid

import pandas as pd

class Explorer(abc.ABC):

    def __init__(self, model, landscape, rounds, experiment_budget, query_budget, initial_sequences):
        self.model = model
        self.landscape = landscape
        self.rounds = rounds
        self.experiment_budget = experiment_budget
        self.query_budget = query_budget
        self.initial_sequences = initial_sequences
        self.run_id = str(uuid.uuid1())

    @abc.abstractmethod
    def propose_sequences(self, batches):
        pass


    def run(self, verbose=True):
        """Run the exporer."""

        sequences = pd.DataFrame({
            'sequence': self.initial_sequences,
            'true_score': self.landscape.get_fitness(self.initial_sequences),
            'model_score': self.landscape.get_fitness(self.initial_sequences),
            'round': 0, 
            'measurement_cost' : len(self.initial_sequences),
            'run_id': self.run_id
        })

        for r in range(1, self.rounds + 1):
            self.model.train(sequences['sequence'].to_numpy(), sequences['ground_truth'].to_numpy())

            seqs, preds = self.propose_sequences(sequences)
            ground_truth = self.landscape.get_fitness(seqs)

            if len(seqs) > self.experiment_budget:
                raise ValueError('Must propose <= `self.experiment_budget` sequences per round')

            sequences = sequences.append(
                pd.DataFrame({
                    'sequence': seqs,
                    'model_score': preds,
                    'true_score': ground_truth,
                    'round': r, 
                    'measurement_cost' : int(r * len(seqs)),
                    'run_id': self.run_id
                })
            )

            if verbose:
                print(
                    f"round: {r}, top: {ground_truth.max()}"
                )

        return sequences
