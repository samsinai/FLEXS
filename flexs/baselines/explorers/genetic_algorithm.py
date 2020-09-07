import numpy as np
import torch

import flexs
import flexs.utils.sequence_utils as s_utils


class GeneticAlgorithm(flexs.Explorer):
    """A genetic algorithm explorer with single point mutations and no crossover.

    Based on the `parent_selection_strategy`, this class implements one of three
    genetic algorithms:

        1. If `parent_selection_strategy == 'top-k'`, we have a traditional
           genetic algorithm where the top-k scoring sequences in the
           population become parents.

        2. If `parent_selection_strategy == 'softmax-sampling'`, we have a
           genetic algorithm based off of the Wright-Fisher model of evolution,
           where members of the population become parents with a probability
           exponential to their fitness (softmax the scores then sample).

    If `recombination_strategy` is None (the default), no recombination happens,
    and children are simply mutated versions of parents.
    If `recombination_strategy` is not None, children are generated through recombining
    parents according to the strategy (currently, '1-point-crossover' and
    `n-tile-crossover` are supported).

    """

    def __init__(
        self,
        model,
        landscape,
        rounds,
        initial_sequence_data,
        experiment_budget,
        query_budget,
        alphabet,
        population_size: int,
        parent_selection_strategy: str,
        children_proportion: float,
        log_file=None,
        parent_selection_proportion: float = None,
        softmax_temperature: float = None,
        recombination_strategy: str = None,
        avg_crossovers: int = None,
        num_crossover_tiles: int = None,
        seed: int = None,
    ):
        name = f"GeneticAlgorithm_pop_size={population_size}_parents={parent_selection_strategy}_recomb={recombination_strategy}"

        super().__init__(
            model,
            landscape,
            name,
            rounds,
            experiment_budget,
            query_budget,
            initial_sequence_data,
            log_file,
        )
        self.alphabet = alphabet
        self.population_size = population_size

        # Validate parent_selection_strategy
        valid_parent_selection_strategies = ["top-proportion", "softmax-sampling"]
        if parent_selection_strategy not in valid_parent_selection_strategies:
            raise ValueError(
                f"parent_selection_strategy must be one of {valid_parent_selection_strategies}"
            )
        if (
            parent_selection_strategy == "top-proportion"
            and parent_selection_proportion is None
        ):
            raise ValueError(
                "if top-proportion, parent_selection_proportion cannot be None"
            )
        if (
            parent_selection_strategy == "softmax-sampling"
            and softmax_temperature is None
        ):
            raise ValueError("if softmax-sampling, softmax_temperature cannot be None")
        self.parent_selection_strategy = parent_selection_strategy
        self.softmax_temperature = softmax_temperature

        self.children_proportion = children_proportion
        self.parent_selection_proportion = parent_selection_proportion

        # Validate recombination_strategy
        valid_recombination_strategies = [None, "1-point-crossover", "n-tile-crossover"]
        if recombination_strategy not in valid_recombination_strategies:
            raise ValueError(
                f"recombination_strategy must be one of {valid_recombination_strategies}"
            )
        if recombination_strategy == "n-tile-crossover" and (
            avg_crossovers is None or num_crossover_tiles is None
        ):
            raise ValueError(
                "if n-tile-crossover, avg_crossovers and num_crossover_tiles cannot be None"
            )
        self.recombination_strategy = recombination_strategy
        self.avg_crossovers = avg_crossovers
        self.num_crossover_tiles = num_crossover_tiles

        self.rng = np.random.default_rng(seed)

    def _choose_parents(self, scores, num_parents):
        """Return parent indices according to `self.parent_selection_strategy`."""

        if self.parent_selection_strategy == "top-proportion":
            k = int(self.parent_selection_proportion * self.population_size)
            return self.rng.choice(np.argsort(scores)[-k:], num_parents)

        elif self.parent_selection_strategy == "softmax-sampling":
            fitnesses = np.exp(scores / self.softmax_temperature)
            probs = torch.Tensor(fitnesses / np.sum(fitnesses))
            return torch.multinomial(probs, num_parents, replacement=True).numpy()

    def _recombine(self, pop, scores):
        num_children = int(self.children_proportion * self.population_size)

        if self.recombination_strategy is None:
            parents = self._choose_parents(scores, num_children)
            return pop[parents]

        """elif self.recombination_strategy == "1-point-crossover":
            parents = self._choose_parents(
                pop, scores, argsorted_scores, 2 * num_children
            )

            children = np.empty((num_children, pop.shape[1]))
            crossover_points = self.rng.integers(1, pop.shape[1], num_children)

            # Didn't find a way to vectorize this function, but it's not a
            # huge deal since performance analysis shows that the predictor
            # is the bottleneck, not these loops here.
            for i in range(num_children):
                children[i, : crossover_points[i]] = pop[
                    parents[2 * i], : crossover_points[i]
                ]
                children[i, crossover_points[i] :] = pop[
                    parents[2 * i + 1], crossover_points[i] :
                ]

            return children

        elif self.recombination_strategy == "n-tile-crossover":
            parents = self._choose_parents(pop, scores, argsorted_scores, num_children)

            children = pop[parents].copy()
            crossover_prob_per_tile = self.avg_crossovers / self.num_crossover_tiles

            # indices = list of tile boundary points (n + 1 points for n tiles)
            indices = list(
                range(0, pop.shape[1], pop.shape[1] // self.num_crossover_tiles)
            ) + [pop.shape[1]]

            # Replace current tile with the corresponding tile from a random parent w/ some probability
            for i in range(num_children):
                for j in range(self.num_crossover_tiles):
                    if self.rng.random() <= crossover_prob_per_tile:
                        children[i, indices[j] : indices[j + 1]] = pop[
                            self.rng.choice(parents), indices[j] : indices[j + 1]
                        ]

            return children"""

    def propose_sequences(self, measured_sequences):
        """Run genetic algorithm explorer."""

        # Set the torch seed by generating a random integer from the pre-seeded self.rng
        torch.manual_seed(self.rng.integers(-(2 ** 31), 2 ** 31))

        measured_sequence_set = set(measured_sequences["sequence"])

        # Create initial population by choosing parents from `measured_sequences`
        initial_pop_inds = self._choose_parents(
            measured_sequences["true_score"].to_numpy(), self.population_size,
        )
        pop = measured_sequences["sequence"].to_numpy()[initial_pop_inds]
        scores = measured_sequences["true_score"].to_numpy()[initial_pop_inds]

        sequences = {}
        initial_cost = self.model.cost
        while self.model.cost - initial_cost + self.population_size < self.query_budget:
            # Create "children" by recombining parents selected from population
            # according to self.parent_selection_strategy and self.recombination_strategy
            parents = self._recombine(pop, scores)

            # Single-point mutation of children (for now)
            children = []
            for seq in parents:
                child = s_utils.generate_random_mutant(seq, 1 / len(seq), self.alphabet)

                if child not in measured_sequence_set and child not in sequences:
                    children.append(child)

            if len(children) == 0:
                continue

            children = np.array(children)
            child_scores = self.model.get_fitness(children)

            # Now kick out the worst samples and replace them with the new children
            argsorted_scores = np.argsort(scores)
            pop[argsorted_scores[: len(children)]] = children
            scores[argsorted_scores[: len(children)]] = child_scores

            sequences.update(zip(children, child_scores))

        # We propose the top `self.experiment_budget` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.experiment_budget : -1]

        return new_seqs[sorted_order], preds[sorted_order]
