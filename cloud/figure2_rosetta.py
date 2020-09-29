"""Get data for Rosetta figure in paper."""
import flexs
from flexs import baselines
import flexs.utils.sequence_utils as s_utils
from typing import Callable


def run_explorer(explorer, landscape, wt, start_name):
    alphabet = s_utils.AAS
    sequences_batch_size = 100
    model_queries_per_batch = 2000
    model = flexs.LandscapeAsModel(landscape)

    if explorer == "adalead":
        exp = baselines.explorers.Adalead(
            model,
            rounds=10,
            recomb_rate=0.2,
            starting_sequence=wt,
            sequences_batch_size=sequences_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            alphabet=alphabet,
            log_file=f"runs/3msi/{explorer}_start={start_name}.csv",
        )

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

        exp = baselines.explorers.CbAS(
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
            log_file=f"runs/3msi/{explorer}_start={start_name}.csv",
        )

    elif explorer == "cmaes":

        exp = baselines.explorers.CMAES(
            model,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=sequences_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            alphabet=alphabet,
            population_size=40,
            max_iter=400,
            log_file=f"runs/3msi/{explorer}_start={start_name}.csv",
        )

    elif explorer == "dynappo":

        exp = baselines.explorers.DynaPPO(
            model=model,
            landscape=landscape,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=sequences_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            num_experiment_rounds=10,
            num_model_rounds=8,
            alphabet=alphabet,
            log_file=f"runs/3msi/{explorer}_start={start_name}.csv",
        )

    elif explorer == "genetic":

        exp = baselines.explorers.GeneticAlgorithm(
            model=model,
            population_size=40,
            parent_selection_strategy="wright-fisher",
            beta=5,
            children_proportion=0.3,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=sequences_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            alphabet=alphabet,
            log_file=f"runs/3msi/{explorer}_start={start_name}.csv",
        )

    elif explorer == "random":

        exp = baselines.explorers.Random(
            model=model,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=sequences_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            alphabet=alphabet,
            log_file=f"runs/3msi/{explorer}_start={start_name}.csv",
        )

    exp.run(landscape)


def main():
    problem = flexs.landscapes.rosetta.registry()["3msi"]
    landscape = flexs.landscapes.RosettaFolding(
        **problem["params"], sigmoid_center=-6, sigmoid_norm_value=12
    )

    for explorer in ["cmaes", "adalead", "genetic", "cbas", "dbas", "dynappo"]:
        for start_name, start_seq in problem["starts"].items():
            print(f"\n{explorer}, start {start_name}\n")
            run_explorer(explorer, landscape, start_seq, start_name)


if __name__ == "__main__":
    main()
