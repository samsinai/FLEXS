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
from explorers.dqn_explorer import DQN_Explorer

LANDSCAPE_TYPES_RNA = {"RNA": [0], "TF": []}
dqn_explorer_prod = DQN_Explorer(virtual_screen=20)
dqn_explorer_prod.debug = False
evaluator_dqn = Evaluator(
    dqn_explorer_prod,
    landscape_types=LANDSCAPE_TYPES_RNA,
    path="../simulations/RNA/eval_DQN/",
)
evaluator_dqn.evaluate_for_landscapes(evaluator_dqn.efficiency, num_starts=3)
