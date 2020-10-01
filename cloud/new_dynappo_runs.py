import argparse
import flexs
from flexs import baselines
import flexs.utils.sequence_utils as s_utils

sequences_batch_size = 100
model_queries_per_batch = 2000

def run_explorer_TF(landscape, wt, problem_name, start_num):
    alphabet = s_utils.DNAA
    def make_explorer(model, ss):
        return baselines.explorers.DynaPPO(
            landscape=landscape,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=sequences_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            num_experiment_rounds=10,
            num_model_rounds=8,
            env_batch_size=4,
            alphabet=alphabet,
            log_file=f"runs/new_dynappo/{problem_name}_start{start_num}_ss{ss}"
        )
    results = flexs.evaluate.robustness(landscape, make_explorer, verbose=False)
    return results

def run_explorer_RNA(landscape, wt, problem_name, start_num):
    alphabet = s_utils.RNAA
    def make_explorer(model, ss):
        return baselines.explorers.DynaPPO(
            landscape=landscape,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=sequences_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            num_experiment_rounds=10,
            num_model_rounds=8,
            env_batch_size=4,
            alphabet=alphabet,
            log_file=f"runs/new_dynappo/{problem_name}_start{start_num}_ss{ss}"
        )
    results = flexs.evaluate.robustness(landscape, make_explorer, verbose=False)
    return results

task = "TF"

if task == "TF":
    for p in ["SIX6_REF_R1", "POU3F4_REF_R1", "PAX3_G48R_R1", "VAX2_REF_R1", "VSX1_REF_R1"]:
        problem = flexs.landscapes.tf_binding.registry()[p]
        landscape = flexs.landscapes.TFBinding(**problem["params"])
        for s in range(13):
            wt = problem["starts"][s]
            results = run_explorer_TF(landscape, wt, p, s)
elif task == "RNA":
    for p in ["L14_RNA1", "L14_RNA1+2", "C21_L100_RNA1+3"]:
        problem = flexs.landscapes.rna.registry()[p]
        landscape = flexs.landscapes.RNABinding(**problem["params"])
        for s in range(5):
            wt = problem["starts"][s]
            results = run_explorer_RNA(landscape, wt, p, s)
