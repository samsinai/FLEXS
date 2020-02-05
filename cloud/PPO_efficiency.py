import os
import sys
sys.path.append('../')
import RNA 
from utils.sequence_utils import generate_random_mutant
from utils.model_architectures import Linear, NLNN, CNNa
from models.Noisy_models.Neural_network_models import NN_model
from models.Ground_truth_oracles.RNA_landscape_models import RNA_landscape_constructor
from models.Noisy_models.Ensemble import Ensemble_models
from evaluators.Evaluator import Evaluator
from models.Ground_truth_oracles.TF_binding_landscape_models import *
from explorers.PPO_explorer import PPO_explorer

LANDSCAPE_TYPES_RNA = {"RNA" : [0,1,12,20,25,31], "TF": []}
LANDSCAPE_TYPES_TF = {"RNA": [], "TF": ['POU3F4_REF_R1','PAX3_G48R_R1','SIX6_REF_R1', 'VAX2_REF_R1', 'VSX1_REF_R1']}

ppo_explorer = PPO_explorer(batch_size=100, virtual_screen=20)
ppo_explorer.debug = False

os.makedirs("../simulations/eval/RNA/PPO/", exist_ok=True)
os.makedirs("../simulations/eval/TF/PPO/", exist_ok=True)
evaluator_ppo = Evaluator(ppo_explorer, landscape_types=LANDSCAPE_TYPES_RNA, path="../simulations/eval/RNA/PPO/")
evaluator_ppo.evaluate_for_landscapes(evaluator_ppo.efficiency, num_starts=3)
evaluator_ppo = Evaluator(ppo_explorer, landscape_types=LANDSCAPE_TYPES_TF, path="../simulations/eval/TF/PPO/")
evaluator_ppo.evaluate_for_landscapes(evaluator_ppo.efficiency, num_starts=5)