import flexs
from flexs import baselines


def robustness_consistency(landscape, make_explorer, signal_strengths=[0, 0.5, 0.75, 0.9, 1]):
    """Evaluates explorer outputs as a function of the noisyness of its model."""

    results = []
    for ss in signal_strengths:
        print(f"Evaluating for robustness and consistency with model accuracy; signal_strength: {ss}")

        model = baselines.models.NoisyAbstractModel(landscape, signal_strength=ss)
        explorer = make_explorer(model, ss)
        res = explorer.run()

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
        res = explorer.run() #TODO: is this being logged? because the last budget pair would take very long

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
        res = explorer.run()

        results.append((rounds, res))

    return results
