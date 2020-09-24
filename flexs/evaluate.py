"""A small set of evaluation metrics to benchmark explorers."""
from typing import Callable, List

import flexs
from flexs import baselines


def robustness(
    landscape: flexs.Landscape,
    make_explorer: Callable[[flexs.Model, float], flexs.Explorer],
    signal_strengths: List[float] = [0, 0.5, 0.75, 0.9, 1],
):
    """
    Evaluate explorer outputs as a function of the noisyness of its model.

    It runs the same explorer with `flexs.NoisyAbstractModel`'s of different signal strengths.

    Args:
        landscape: The landscape to run on.
        make_explorer: A function that takes in
            a model and signal strength (for potential bookeeping/logging purposes) and
            returns the desired explorer.
        signal_strengths: A list of signal strengths between 0 and 1.
    """
    results = []
    for ss in signal_strengths:
        print(f"Evaluating for robustness with model accuracy; signal_strength: {ss}")

        model = baselines.models.NoisyAbstractModel(landscape, signal_strength=ss)
        explorer = make_explorer(model, ss)
        res = explorer.run(landscape)

        results.append((ss, res))

    return results


def efficiency(
    landscape,
    make_explorer,
    budgets=[(100, 500), (100, 5000), (1000, 5000), (1000, 10000)],
):
    """
    Evaluates explorer outputs as a function of the number of allowed
    ground truth measurements and model queries per round.

    """

    results = []
    for sequences_batch_size, model_queries_per_batch in budgets:
        print(
            f"Evaluating for sequences_batch_size: {sequences_batch_size}, model_queries_per_batch: {model_queries_per_batch}"
        )
        explorer = make_explorer(sequences_batch_size, model_queries_per_batch)
        res = explorer.run(
            landscape
        )  # TODO: is this being logged? because the last budget pair would take very long

        results.append(((sequences_batch_size, model_queries_per_batch), res))

    return results


def adaptivity(
    landscape,
    make_explorer,
    num_rounds=[1, 10, 100, 1000],
    total_ground_truth_measurements=1000,
    total_model_queries=5000,
):
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
