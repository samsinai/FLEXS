"""Script to compare different versions (mutative vs. constructive) versions of DyNA-PPO. For reference, the original DyNA-PPO paper is constructive."""

import flexs
from flexs import baselines
import flexs.utils.sequence_utils as s_utils
from typing import Callable

alphabet = s_utils.RNAA
sequences_batch_size = 10
model_queries_per_batch = 200

def run_dynappo_constructive(landscape, wt, problem_name, start_num):
    def make_explorer(model, ss):
        return baselines.explorers.DynaPPO(
            landscape=landscape,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=sequences_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            num_experiment_rounds=10,
            num_model_rounds=8,
            alphabet=alphabet,
            log_file=f"runs/dynappo_constructive/{problem_name}_start{start_num}_ss{ss}"
        )
    results = flexs.evaluate.robustness(landscape, make_explorer, verbose=False)
    return results

def run_dynappo_mutative(landscape, wt, problem_name, start_num):
    def make_explorer(model, ss):
        return baselines.explorers.DynaPPOMutative(
            landscape=landscape,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=sequences_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            num_experiment_rounds=10,
            num_model_rounds=8,
            alphabet=alphabet,
            log_file=f"runs/dynappo_mutative/{problem_name}_start{start_num}_ss{ss}"
        )
    results = flexs.evaluate.robustness(landscape, make_explorer, verbose=False)
    return results

if __name__ == "__main__":
    for p in ["L14_RNA1"]:
        problem = flexs.landscapes.rna.registry()[p]
        landscape = flexs.landscapes.RNABinding(**problem["params"])
        for s in range(5):
            wt = problem["starts"][s]
            results = run_dynappo_constructive(landscape, wt, p, s)
            results = run_dynappo_mutative(landscape, wt, p, s)
