import sys

sys.path.append("../")
import RNA
from utils.model_architectures import Linear, NLNN, CNNa
from models.Noisy_models.Neural_network_models import NN_model
from models.Ground_truth_oracles.RNA_landscape_models import RNA_landscape_constructor
from models.Noisy_models.Ensemble import Ensemble_models
from evaluators.Evaluator import Evaluator
from models.Ground_truth_oracles.TF_binding_landscape_models import *
from explorers.elitist_explorers import XE_IS

LANDSCAPE_TYPES = {"GFP": [1, 5, 9]}
xeis_explorer_prod = XE_IS(recomb_rate=0.2, virtual_screen=20)
xeis_explorer_prod.debug = False
evaluator_xeis = Evaluator(
    xeis_explorer_prod,
    landscape_types=LANDSCAPE_TYPES,
    path="../simulations/eval_xeis_consistency/",
)
evaluator_xeis.evaluate_for_landscapes(
    evaluator_xeis.consistency_robustness_independence, num_starts=3
)
