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
from explorers.bo_explorer import BO_Explorer
from explorers.dqn_explorer import DQN_Explorer 

LANDSCAPE_TYPES_TF = {"RNA": [], 
                      "TF": [
                                "POU3F4_REF_R1",
                                "PAX3_G48R_R1",
                                "SIX6_REF_R1",
                                "VAX2_REF_R1",
                                "VSX1_REF_R1",
                            ]}

import copy
import os
import sys
import random
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from explorers.base_explorer import Base_explorer
from utils.sequence_utils import *

import numpy as np
from bisect import bisect_left

from explorers.new_bo_explorer import New_BO_Explorer
    
bo_explorer_prod = New_BO_Explorer()
bo_explorer_prod.debug = False

evaluator_bo=Evaluator(bo_explorer_prod,landscape_types=LANDSCAPE_TYPES_TF, path="../simulations/eval/")
evaluator_bo.evaluate_for_landscapes(evaluator_bo.consistency_robustness_independence, num_starts=5)