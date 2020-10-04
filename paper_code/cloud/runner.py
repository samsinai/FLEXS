import argparse

import flexs
from flexs import baselines
from flexs.utils import sequence_utils as s_utils


def run_explorer_robustness(args, landscape, wt, problem_name, start_num):
    alphabet = s_utils.RNAA if args.landscapes == "rna" else s_utils.DNAA
    if args.explorer == "adalead":

        def make_explorer(model, ss):
            return baselines.explorers.Adalead(
                model,
                rounds=10,
                recomb_rate=0.2,
                starting_sequence=wt,
                sequences_batch_size=args.sequences_batch_size,
                model_queries_per_batch=args.model_queries_per_batch,
                alphabet=alphabet,
                log_file=f"runs/{args.explorer}/{args.landscapes}_{problem_name}_start{start_num}_ss{ss}",
            )

        results = flexs.evaluate.robustness(landscape, make_explorer, verbose=False)
    elif args.explorer == "cbas" or args.explorer == "dbas":
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

        def make_explorer(model, ss):
            return baselines.explorers.CbAS(
                model,
                generator=g,
                rounds=10,
                Q=0.8,
                algo=args.explorer,
                starting_sequence=wt,
                sequences_batch_size=args.sequences_batch_size,
                model_queries_per_batch=args.model_queries_per_batch,
                mutation_rate=2.0 / len(wt),
                alphabet=alphabet,
                log_file=f"runs/{args.explorer}/{args.landscapes}_{problem_name}_start{start_num}_ss{ss}",
            )

        results = flexs.evaluate.robustness(landscape, make_explorer, verbose=False)
    elif args.explorer == "cmaes":

        def make_explorer(model, ss):
            return baselines.explorers.CMAES(
                model,
                rounds=10,
                starting_sequence=wt,
                sequences_batch_size=args.sequences_batch_size,
                model_queries_per_batch=args.model_queries_per_batch,
                alphabet=alphabet,
                population_size=40,
                max_iter=400,
                log_file=f"runs/{args.explorer}/{args.landscapes}_{problem_name}_start{start_num}_ss{ss}",
            )

        results = flexs.evaluate.robustness(landscape, make_explorer, verbose=False)
    elif args.explorer == "dynappo":

        def make_explorer(model, ss):
            return baselines.explorers.DynaPPO(
                landscape=landscape,
                rounds=10,
                starting_sequence=wt,
                sequences_batch_size=args.sequences_batch_size,
                model_queries_per_batch=args.model_queries_per_batch,
                num_experiment_rounds=10,
                num_model_rounds=8,
                alphabet=alphabet,
                log_file=f"runs/{args.explorer}/{args.landscapes}_{problem_name}_start{start_num}_ss{ss}",
            )

        results = flexs.evaluate.robustness(landscape, make_explorer, verbose=False)
    return results


def run_all(args):
    if args.landscapes == "rna":
        for p in ["L14_RNA1", "L14_RNA1+2"]:
            problem = flexs.landscapes.rna.registry()[p]
            landscape = flexs.landscapes.RNABinding(**problem["params"])
            for s in range(5):
                wt = problem["starts"][s]
                results = run_explorer_robustness(args, landscape, wt, p, s)
    elif args.landscapes == "tf":
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
                results = run_explorer_robustness(args, landscape, wt, p, s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--explorer",
        choices=["adalead", "cbas", "dbas", "cmaes", "dynappo"],
        required=True,
    )
    parser.add_argument("--landscapes", choices=["rna", "tf"], required=True)
    parser.add_argument("--sequences_batch_size", type=int, default=100)
    parser.add_argument("--model_queries_per_batch", type=int, default=2000)

    args = parser.parse_args()

    run_all(args)
