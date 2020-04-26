from meta.model import Model


class Multi_dimensional_model(Model):
    """Build a composite of ground truth models to make more complicated landscapes."""

    def __init__(self, models, combined_func=sum):
        self.sequences = {}
        self.models = [model for model in models]
        self.dim = len(self.models)
        self.model_names = [str(model) for model in self.models]
        self.combined_func = combined_func

        self.expanded_sequences = {}

    def populate_model(self, data=None):
        for sequence in data:
            self.sequences[sequence] = tuple(
                [model.get_fitness(sequence) for model in self.models]
            )

    def _fitness_function(self, sequence):
        return self.combined_func(
            [model.get_fitness(sequence) for model in self.models]
        )

    def get_fitness(self, sequence):
        if sequence in self.sequences:
            return self.sequences[sequence]
        else:
            self.sequences[sequence] = self._fitness_function(sequence)
            self.expanded_sequences[sequence] = list(
                [model.get_fitness(sequence) for model in self.models]
            )
            return self.sequences[sequence]

    def break_down_fitness_and_sort(self):
        sorted_by_value = sorted(
            self.sequences.items(), key=lambda kv: kv[1], reverse=True
        )
        output = []
        for sequence, fitness in sorted_by_value:
            output.append(self.expanded_sequences[sequence] + [fitness])
        return output
