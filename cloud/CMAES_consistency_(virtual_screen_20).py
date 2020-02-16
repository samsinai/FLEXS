import os
import sys

sys.path.append("../")
import RNA
from utils.sequence_utils import generate_random_mutant
from utils.model_architectures import Linear, NLNN, CNNa
from models.Noisy_models.Neural_network_models import NN_model
from models.Ground_truth_oracles.RNA_landscape_models import RNA_landscape_constructor
from models.Noisy_models.Ensemble import Ensemble_models
from evaluators.Evaluator import Evaluator
from models.Ground_truth_oracles.TF_binding_landscape_models import *
from explorers.CMAES_explorer import CMAES_explorer

LANDSCAPE_TYPES_RNA = {"RNA": [0, 1, 12, 20, 25, 31], "TF": []}
LANDSCAPE_TYPES_TF = {
    "RNA": [],
    "TF": [
        "POU3F4_REF_R1",
        "PAX3_G48R_R1",
        "SIX6_REF_R1",
        "VAX2_REF_R1",
        "VSX1_REF_R1",
    ],
}

cmaes_explorer = CMAES_explorer(batch_size=100, virtual_screen=20)
cmaes_explorer.debug = False

os.makedirs("../simulations/eval/RNA/CMAES/", exist_ok=True)
os.makedirs("../simulations/eval/TF/CMAES/", exist_ok=True)
evaluator_cmaes = Evaluator(
    cmaes_explorer,
    landscape_types=LANDSCAPE_TYPES_RNA,
    path="../simulations/eval/RNA/CMAES/",
)
evaluator_cmaes.evaluate_for_landscapes(
    evaluator_cmaes.consistency_robustness_independence, num_starts=3
)
evaluator_cmaes = Evaluator(
    cmaes_explorer,
    landscape_types=LANDSCAPE_TYPES_TF,
    path="../simulations/eval/TF/CMAES/",
)
evaluator_cmaes.evaluate_for_landscapes(
    evaluator_cmaes.consistency_robustness_independence, num_starts=5
)
