import flexs
from flexs import baselines


def robustness(landscape, make_explorer, signal_strengths=[0, 0.5, 0.8, 0.9, 1]):
    """Evaluates explorer outputs as a function of the noisyness of its model."""

    results = []
    for ss in signal_strengths:
        print(f"Evaluating for robustness to model accuracy; signal_strength: {ss}")

        model = baselines.models.NoisyAbstractModel(landscape, signal_strength=ss)
        explorer = make_explorer(model, ss)
        sequences = explorer.run()

        results.append((ss, sequences))

    return results


def efficiency(
    landscape,
    make_explorer,
    budgets=[(100, 500), (100, 1000), (1000, 5000), (1000, 10000)],
):
    """
    Evaluates explorer outputs as a function of the number of allowed
    ground truth measurements and model queries per round.

    """

    results = []
    for ground_truth_measurements_per_round, model_queries_per_round in budgets:
        print(
            f"Evaluating for ground_truth_measurements_per_round: {ground_truth_measurements_per_round}, model_queries_per_round: {model_queries_per_round}"
        )
        explorer = make_explorer(
            ground_truth_measurements_per_round, model_queries_per_round
        )
        sequences = explorer.run()

        results.append(
            ((ground_truth_measurements_per_round, model_queries_per_round), sequences)
        )

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
        sequences = explorer.run()

        results.append((rounds, sequences))

    return results
