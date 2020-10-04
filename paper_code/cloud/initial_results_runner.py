import argparse
import flexs
from flexs import baselines
import flexs.utils.sequence_utils as s_utils

def run_explorer_robustness():
    problem = flexs.landscapes.rna.registry()['L14_RNA1']
    landscape = flexs.landscapes.RNABinding(**problem["params"])
    wt = problem["starts"][0]

    '''# random
    def make_explorer(model, ss):
        return baselines.explorers.Random(
            model,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=100,
            model_queries_per_batch=2000,
            alphabet=s_utils.RNAA,
            log_file=f"runs/initial_results/random/ss{ss}.csv"
        )
    flexs.evaluate.robustness(landscape, make_explorer, verbose=False)

    # genetic algo
    def make_explorer(model, ss):
        return baselines.explorers.GeneticAlgorithm(
            model,
            population_size=20,
            parent_selection_strategy='wright-fisher',
            beta=0.1,
            children_proportion=0.2,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=100,
            model_queries_per_batch=2000,
            alphabet=s_utils.RNAA,
            log_file=f"runs/initial_results/genetic/ss{ss}.csv"
        )
    flexs.evaluate.robustness(landscape, make_explorer, verbose=False)

    # adalead
    def make_explorer(model, ss):
        return baselines.explorers.Adalead(
            model,
            rounds=10,
            recomb_rate=0.2,
            starting_sequence=wt,
            sequences_batch_size=100,
            model_queries_per_batch=2000,
            alphabet=s_utils.RNAA,
            log_file=f"runs/initial_results/adalead/ss{ss}.csv"
        )
    flexs.evaluate.robustness(landscape, make_explorer, verbose=False)

    # cbas
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
        verbose=False,
    )
    def make_explorer(model, ss):
        return baselines.explorers.CbAS(
            model,
            generator=g,
            rounds=10,
            Q=0.8,
            algo="cbas",
            starting_sequence=wt,
            sequences_batch_size=100,
            model_queries_per_batch=2000,
            mutation_rate=2.0/len(wt),
            alphabet=s_utils.RNAA,
            log_file=f"runs/initial_results/cbas/ss{ss}.csv"
        )
    flexs.evaluate.robustness(landscape, make_explorer, verbose=False)

    # cmaes
    def make_explorer(model, ss):
        return baselines.explorers.CMAES(
            model,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=100,
            model_queries_per_batch=2000,
            alphabet=s_utils.RNAA,
            population_size=40,
            max_iter=400,
            log_file=f"runs/initial_results/cmaes/ss{ss}.csv"
        )
    flexs.evaluate.robustness(landscape, make_explorer, verbose=False)'''

    # dynappo
    def make_explorer(model, ss):
        return baselines.explorers.DynaPPO(
            model=model,
            landscape=landscape,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=100,
            model_queries_per_batch=2000,
            num_experiment_rounds=10,
            num_model_rounds=8,
            alphabet=s_utils.RNAA,
            log_file=f"runs/initial_results/dynappo/ss{ss}.csv"
        )
    flexs.evaluate.robustness(landscape, make_explorer, verbose=False)

    # dynappo mutative
    def make_explorer(model, ss):
        return baselines.explorers.DynaPPOMutative(
            model=model,
            landscape=landscape,
            rounds=10,
            starting_sequence=wt,
            sequences_batch_size=100,
            model_queries_per_batch=2000,
            num_experiment_rounds=10,
            num_model_rounds=8,
            alphabet=s_utils.RNAA,
            log_file=f"runs/initial_results/dynappo_mutative/ss{ss}.csv"
        )
    flexs.evaluate.robustness(landscape, make_explorer, verbose=False)

if __name__ == "__main__":
    run_explorer_robustness()