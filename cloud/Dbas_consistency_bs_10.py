import sys

sys.path.append("../")
import RNA
from evaluators.Evaluator import Evaluator
from explorers.CbAS_DbAS_explorers import CbAS_explorer, DbAS_explorer
from explorers.elitist_explorers import Greedy
from models.Ground_truth_oracles.RNA_landscape_models import RNA_landscape_constructor
from models.Ground_truth_oracles.TF_binding_landscape_models import *
from models.Noisy_models.Ensemble import Ensemble_models
from models.Noisy_models.Neural_network_models import NN_model
from utils.model_architectures import NLNN, VAE, CNNa, Linear

LANDSCAPE_TYPES = {"RNA": [0]}
g = VAE(
    batch_size=10,
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
dbas_explorer = DbAS_explorer(batch_size=10, virtual_screen=20, generator=g)
dbas_explorer.debug = False
evaluator_dbas = Evaluator(
    dbas_explorer,
    landscape_types=LANDSCAPE_TYPES,
    path="../simulations/eval/",
)
evaluator_dbas.evaluate_for_landscapes(
    evaluator_dbas.consistency_robustness_independence, num_starts=5
)
