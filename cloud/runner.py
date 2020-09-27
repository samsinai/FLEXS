import argparse
import flexs
from flexs import baselines
import flexs.utils.sequence_utils as s_utils

def run_explorer_robustness(args, landscape, wt):
    # needs signal strength in name
    # log_file=f'plots/2b/random/run{i}.csv' arg
    if args.explorer == "adalead":
        def make_explorer(model, ss):
            return baselines.explorers.Adalead(
                model,
                rounds=5,
                recomb_rate=0.2,
                starting_sequence=wt,
                sequences_batch_size=100,
                model_queries_per_batch=2000,
                alphabet=s_utils.RNAA
            )
        results = flexs.evaluate.robustness(landscape, make_explorer)
    elif args.explorer == "cbas" or args.explorer == "dbas":
        g = baselines.explorers.cbas_dbas.VAE(
            seq_length=len(wt),
            alphabet=s_utils.RNAA,
            batch_size=100,
            latent_dim=2,
            intermediate_dim=250,
            epochs=10,
            epsilon_std=1.0,
            beta=1,
            validation_split=0,
            min_training_size=100,
            mutation_rate=2,
            verbose=False,
        )
        def make_explorer(model, ss):
            return baselines.explorers.CbAS(
                model,
                generator=g,
                rounds=5,
                Q=0.8,
                algo=args.explorer,
                starting_sequence=wt,
                sequences_batch_size=100,
                model_queries_per_batch=2000,
                alphabet=s_utils.RNAA
            )
            # mutation rate?
        results = flexs.evaluate.robustness(landscape, make_explorer)
    elif args.explorer == "cmaes":
        def make_explorer(model, ss):
            return baselines.explorers.CMAES(
                model,
                rounds=5,
                starting_sequence=wt,
                sequences_batch_size=100,
                model_queries_per_batch=2000,
                alphabet=s_utils.RNAA,
                population_size=40,
                max_iter=400
            )
        results = flexs.evaluate.robustness(landscape, make_explorer)
    elif args.explorer == "dynappo":
        def make_explorer(model, ss):
            return baselines.explorers.DynaPPO(
                model,
                starting_sequence=wt,
                sequences_batch_size=100,
                model_queries_per_batch=2000,
                num_experiment_rounds=10,
                num_model_rounds=8,
                alphabet=s_utils.RNAA
            )
        results = flexs.evaluate.robustness(landscape, make_explorer)
    return results

def misc(args):
    if args.landscapes == "rna":
        for p in ["L14_RNA1", "L14_RNA1+2"]:
            problem = flexs.landscapes.rna.registry()[p]
            landscape = flexs.landscapes.RNABinding(**problem["params"])
            for s in range(5):
                wt = problem["starts"][s]
                results = run_explorer_robustness(args, landscape, wt)
    elif args.landscapes == "tf":
        for p in ["POU3F4_REF_R1", "PAX3_G48R_R1", "SIX6_REF_R1", "VAX2_REF_R1", "VSX1_REF_R1"]:
            problem = flexs.landscapes.tf_binding.registry()[p]
            landscape = flexs.landscapes.TFBinding(**problem["params"])
            for s in range(13):
                wt = problem["starts"][s]
                results = run_explorer_robustness(args, landscape, wt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explorer", choices=["adalead", "cbas", "dbas", "cmaes", "dynappo"])
    parser.add_argument("--landscapes", choices=["rna", "tf"])

    args = parser.parse_args()
