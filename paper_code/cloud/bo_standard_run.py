"""Because BO relies on an ensemble and uncertainty estimates, but the Noisy Abstract Models typically return single values, we wrap everything in an Ensemble and run BO separately."""
from typing import Callable

import flexs
from flexs import baselines
from flexs.utils import sequence_utils as s_utils

sequences_batch_size = 100
model_queries_per_batch = 2000


def run_bo(landscape, wt, problem_name, start_num):
    alphabet = s_utils.DNAA

    def _robustness(
        landscape: flexs.Landscape,
        make_explorer: Callable[[flexs.Model, float, str], flexs.Explorer],
    ):
        results = []

        for ss in [0.0, 0.5, 0.9, 1.0]:
            print(
                f"Evaluating for robustness with model accuracy; signal_strength: {ss}"
            )

            model = flexs.Ensemble(
                [baselines.models.NoisyAbstractModel(landscape, signal_strength=ss)],
                combine_with=lambda x: x,
            )
            explorer = make_explorer(model, ss, tag=f"ss{ss}")
            res = explorer.run(landscape, verbose=False)

            results.append((ss, res))

        print("Evaluating for robustness with model accuracy; using 3xCNN ensemble")
        cnn_ensemble = flexs.Ensemble(
            [
                baselines.models.CNN(
                    len(wt),
                    alphabet=alphabet,
                    num_filters=32,
                    hidden_size=100,
                    loss="MSE",
                )
                for i in range(3)
            ],
            combine_with=lambda x: x,
        )
        explorer = make_explorer(cnn_ensemble, ss, tag="cnn")
        res = explorer.run(landscape, verbose=False)

        results.append((None, res))

        return results

    def make_explorer(model, ss, tag):
        return baselines.explorers.BO_Standard(
            model=model,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=sequences_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            alphabet=alphabet,
            log_file=f"runs/bo_standard/{problem_name}_start{start_num}_{tag}",
        )

    results = _robustness(landscape, make_explorer)
    return results


if __name__ == "__main__":
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
            results = run_bo(landscape, wt, p, s)
