"""Get data for Table 1 in paper. TF landscapes only."""
from typing import Callable

import flexs
from flexs import baselines
from flexs.utils import sequence_utils as s_utils


def _robustness(
    landscape: flexs.Landscape,
    make_explorer: Callable[[flexs.Model, float], flexs.Explorer],
):
    """
    This is essentially the robustness evaluator, but with the models hard-coded.
    """
    results = []

    for ss in [0.0, 1.0]:
        print(f"Evaluating for robustness with model accuracy; signal_strength: {ss}")

        model = baselines.models.NoisyAbstractModel(landscape, signal_strength=ss)
        explorer = make_explorer(model, ss, tag=f"ss{ss}")
        res = explorer.run(landscape, verbose=False)

        results.append((ss, res))

    cnn_ensemble = flexs.Ensemble(
        [
            baselines.models.CNN(
                len(wt),
                alphabet=s_utils.DNAA,
                num_filters=32,
                hidden_size=100,
                loss="MSE",
            )
            for i in range(3)
        ]
    )
    explorer = make_explorer(cnn_ensemble, ss, tag="cnn")
    res = explorer.run(landscape, verbose=False)

    results.append((None, res))

    return results


def run_explorers(explorer, landscape, wt, problem_name, start_num):
    alphabet = s_utils.DNAA
    sequences_batch_size = 100
    model_queries_per_batch = 2000

    if explorer == "adalead":

        def make_explorer(model, ss, tag):
            return baselines.explorers.Adalead(
                model,
                rounds=10,
                recomb_rate=0.2,
                starting_sequence=wt,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                alphabet=alphabet,
                log_file=f"runs/{explorer}/{problem_name}_start{start_num}_{tag}",
            )

        results = _robustness(landscape, make_explorer)
    elif explorer == "cbas" or explorer == "dbas":
        g = baselines.explorers.cbas_dbas.VAE(
            seq_length=len(wt),
            alphabet=alphabet,
            batch_size=100,
            latent_dim=2,
            intermediate_dim=250,
            epochs=10,
            epsilon_std=1.0,
            beta=1,
            validation_split=0,
            verbose=False,
        )

        def make_explorer(model, ss, tag):
            return baselines.explorers.CbAS(
                model,
                generator=g,
                rounds=10,
                Q=0.8,
                algo=explorer,
                starting_sequence=wt,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                mutation_rate=2.0 / len(wt),
                alphabet=alphabet,
                log_file=f"runs/{explorer}/{problem_name}_start{start_num}_{tag}",
            )

        results = _robustness(landscape, make_explorer)
    elif explorer == "cmaes":

        def make_explorer(model, ss, tag):
            return baselines.explorers.CMAES(
                model,
                rounds=10,
                starting_sequence=wt,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                alphabet=alphabet,
                population_size=40,
                max_iter=400,
                log_file=f"runs/{explorer}/{problem_name}_start{start_num}_{tag}",
            )

        results = _robustness(landscape, make_explorer)
    elif explorer == "dynappo":

        def make_explorer(model, ss, tag):
            return baselines.explorers.DynaPPO(
                model=model,
                landscape=landscape,
                rounds=10,
                starting_sequence=wt,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                num_experiment_rounds=10,
                num_model_rounds=8,
                alphabet=alphabet,
                log_file=f"runs/{explorer}/{problem_name}_start{start_num}_{tag}",
            )

        results = _robustness(landscape, make_explorer)
    return results


if __name__ == "__main__":
    for explorer in ["cmaes", "adalead", "cbas", "dbas", "dynappo"]:
        for p in [
            "POU3F4_REF_R1",
            "PAX3_G48R_R1",
            "SIX6_REF_R1",
            "VAX2_REF_R1",
            "VSX1_REF_R1",
        ]:
            problem = flexs.landscapes.tf_binding.registry()[p]
            landscape = flexs.landscapes.TFBinding(**problem["params"])
            for s in range(13):
                wt = problem["starts"][s]
                results = run_explorers(explorer, landscape, wt, p, s)
