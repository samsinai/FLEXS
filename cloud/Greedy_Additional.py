import sys

sys.path.append("../")
import RNA
from evaluators.Evaluator import Evaluator
from explorers.elitist_explorers import Greedy
from models.Ground_truth_oracles.RNA_landscape_models import RNA_landscape_constructor
from models.Ground_truth_oracles.TF_binding_landscape_models import *
from models.Noisy_models.Ensemble import Ensemble_models
from models.Noisy_models.Neural_network_models import NN_model
from utils.model_architectures import NLNN, CNNa, Linear

LANDSCAPE_TYPES = {"RNA": [i for i in range(31 + 1) if i not in [0, 1, 12, 20, 25, 31]]}
xeis_explorer_prod = Greedy(recomb_rate=0.2, virtual_screen=20)
xeis_explorer_prod.debug = False
evaluator_xeis = Evaluator(
    xeis_explorer_prod,
    landscape_types=LANDSCAPE_TYPES,
    path="../simulations/eval_greedy_consistency/",
)
evaluator_xeis.evaluate_for_landscapes(evaluator_xeis.efficiency, num_starts=3)
