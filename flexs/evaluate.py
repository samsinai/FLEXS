"""A small set of evaluation metrics to benchmark explorers."""
from typing import Callable, List, Tuple

import flexs
from flexs import baselines


def robustness(
    landscape: flexs.Landscape,
    make_explorer: Callable[[flexs.Model, float], flexs.Explorer],
    signal_strengths: List[float] = [0, 0.5, 0.75, 0.9, 1],
    verbose: bool = True,
):
    """
    Evaluate explorer outputs as a function of the noisyness of its model.

    It runs the same explorer with `flexs.NoisyAbstractModel`'s of different
    signal strengths.

    Args:
        landscape: The landscape to run on.
        make_explorer: A function that takes in a model and signal strength
            (for potential bookkeeping/logging purposes) and an explorer.
        signal_strengths: A list of signal strengths between 0 and 1.

    """
    results = []
    for ss in signal_strengths:
        print(f"Evaluating for robustness with model accuracy; signal_strength: {ss}")

        model = baselines.models.NoisyAbstractModel(landscape, signal_strength=ss)
        explorer = make_explorer(model, ss)
        res = explorer.run(landscape, verbose=verbose)

        results.append((ss, res))

    return results


def efficiency(
    landscape: flexs.Landscape,
    make_explorer: Callable[[int, int], flexs.Explorer],
    budgets: List[Tuple[int, int]] = [
        (100, 500),
        (100, 5000),
        (1000, 5000),
        (1000, 10000),
    ],
):
    """
    Evaluate explorer outputs as a function of the number of allowed ground truth
    measurements and model queries per round.

    Args:
        landscape: Ground truth fitness landscape.
        make_explorer: A function that takes in a `sequences_batch_size` and
            a `model_queries_per_batch` and returns an explorer.
        budgets: A list of tuples (`sequences_batch_size`, `model_queries_per_batch`).

    """
    results = []
    for sequences_batch_size, model_queries_per_batch in budgets:
        print(
            f"Evaluating for sequences_batch_size: {sequences_batch_size}, "
            f"model_queries_per_batch: {model_queries_per_batch}"
        )
        explorer = make_explorer(sequences_batch_size, model_queries_per_batch)
        res = explorer.run(
            landscape
        )  # TODO: is this being logged? bc the last budget pair would take very long

        results.append(((sequences_batch_size, model_queries_per_batch), res))

    return results


def adaptivity(
    landscape: flexs.Landscape,
    make_explorer: Callable[[int, int, int], flexs.Explorer],
    num_rounds: List[int] = [1, 10, 100],
    total_ground_truth_measurements: int = 1000,
    total_model_queries: int = 10000,
):
    """
    For a fixed total budget of ground truth measurements and model queries,
    run with different numbers of rounds.

    Args:
        landscape: Ground truth fitness landscape.
        make_explorer: A function that takes in a number of rounds, a
            `sequences_batch_size` and a `model_queries_per_batch` and returns an
            explorer.
        num_rounds: A list of number of rounds to run the explorer with.
        total_ground_truth_measurements: Total number of ground truth measurements
            across all rounds (`sequences_batch_size * rounds`).
        total_model_queries: Total number of model queries across all rounds
            (`model_queries_per_round * rounds`).

    """
    results = []
    for rounds in num_rounds:
        print(f"Evaluating for num_rounds: {rounds}")
        explorer = make_explorer(
            rounds,
            int(total_ground_truth_measurements / rounds),
            int(total_model_queries / rounds),
        )
        res = explorer.run(landscape)

        results.append((rounds, res))

    return results
