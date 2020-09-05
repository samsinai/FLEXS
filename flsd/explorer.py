import abc

import pandas as pd

class Explorer(abc.ABC):

    def __init__(self, model, landscape, rounds, experiment_budget, query_budget, initial_sequences):
        self.model = model
        self.landscape = landscape
        self.rounds = rounds
        self.experiment_budget = experiment_budget
        self.query_budget = query_budget
        self.initial_sequences = initial_sequences

    @abc.abstractmethod
    def propose_sequences(self, batches):
        pass

    def run(self, verbose=True):
        """Run the exporer."""

        sequences = pd.DataFrame({
            'sequence': self.initial_sequences,
            'ground_truth': self.landscape.get_fitness(self.initial_sequences),
            'round': 0
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
                    'pred': preds,
                    'ground_truth': ground_truth,
                    'round': r
                })
            )

            if verbose:
                print(
                    f"round: {r}, top: {ground_truth.max()}"
                )

        return sequences
