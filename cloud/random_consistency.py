import sys

sys.path.append("../")
import RNA
from evaluators.Evaluator import Evaluator
from explorers.random_explorer import Random_explorer
from models.Ground_truth_oracles.RNA_landscape_models import RNA_landscape_constructor
from models.Ground_truth_oracles.TF_binding_landscape_models import *
from models.Noisy_models.Ensemble import Ensemble_models
from models.Noisy_models.Neural_network_models import NN_model
from utils.model_architectures import NLNN, CNNa, Linear
from utils.sequence_utils import generate_random_mutant

LANDSCAPE_TYPES = {
    "RNA": [0, 1, 12, 20, 25, 31],
    "TF": [
        "POU3F4_REF_R1",
        "PAX3_G48R_R1",
        "SIX6_REF_R1",
        "VAX2_REF_R1",
        "VSX1_REF_R1",
    ],
}
random_explorer_prod = Random_explorer(0.05, virtual_screen=20)
random_explorer_prod.debug = False
evaluator_random = Evaluator(
    random_explorer_prod,
    landscape_types=LANDSCAPE_TYPES,
    path="../simulations/eval_random_consistency/",
)
evaluator_random.evaluate_for_landscapes(
    evaluator_random.consistency_robustness_independence, num_starts=15
)
