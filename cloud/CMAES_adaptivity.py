import os
import sys

sys.path.append("../")
import RNA
from evaluators.Evaluator import Evaluator
from explorers.CMAES_explorer import CMAES_explorer
from models.Ground_truth_oracles.RNA_landscape_models import RNA_landscape_constructor
from models.Ground_truth_oracles.TF_binding_landscape_models import *
from models.Noisy_models.Ensemble import Ensemble_models
from models.Noisy_models.Neural_network_models import NN_model
from utils.model_architectures import NLNN, CNNa, Linear
from utils.sequence_utils import generate_random_mutant

LANDSCAPE_TYPES_RNA = {"RNA": [0], "TF": []}

cmaes_explorer = CMAES_explorer(batch_size=100, virtual_screen=20)
cmaes_explorer.debug = False

os.makedirs("../simulations/eval/RNA/CMAES/", exist_ok=True)
evaluator_cmaes = Evaluator(
    cmaes_explorer,
    landscape_types=LANDSCAPE_TYPES_RNA,
    path="../simulations/eval/RNA/CMAES/",
)
evaluator_cmaes.evaluate_for_landscapes(evaluator_cmaes.adaptivity, num_starts=3)
