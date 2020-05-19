import sys

sys.path.append("../")
import RNA
from utils.sequence_utils import generate_xeis_mutant
from utils.model_architectures import Linear, NLNN, CNNa
from models.Noisy_models.Neural_network_models import NN_model
from models.Ground_truth_oracles.RNA_landscape_models import RNA_landscape_constructor
from models.Noisy_models.Ensemble import Ensemble_models
from evaluators.Evaluator import Evaluator
from models.Ground_truth_oracles.TF_binding_landscape_models import *
from explorers.elitist_explorers import XE_IS

LANDSCAPE_TYPES = {"RNA": [0, 1, 12, 20, 25, 31], 
    "TF": [
        "POU3F4_REF_R1",
        "PAX3_G48R_R1",
        "SIX6_REF_R1",
        "VAX2_REF_R1",
        "VSX1_REF_R1",
    ],
}
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
