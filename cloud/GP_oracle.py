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

LANDSCAPE_TYPES_RNA = {"RNA": [0, 12]}
LANDSCAPE_TYPES_TF = {
    "TF": [
        "POU3F4_REF_R1",
        "PAX3_G48R_R1",
        "SIX6_REF_R1",
        "VAX2_REF_R1",
        "VSX1_REF_R1",
    ]
}

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

from explorers.CbAS_DbAS_explorers import CbAS_explorer, DbAS_explorer
from explorers.CMAES_explorer import CMAES_explorer
from explorers.DynaPPO_explorer import DynaPPO_explorer
from explorers.elitist_explorers import Greedy

import tensorflow as tf

pairs = [
    ("CbAS", CbAS_explorer),
    ("DbAS", DbAS_explorer),
    ("CMAES", CMAES_explorer),
    ("DynaPPO", DynaPPO_explorer),
    ("AdaLead", Greedy)
]

# For RNA
for name, exp_fn in pairs:
    if name == "AdaLead":
        explorer = exp_fn(batch_size=100, virtual_screen=20, recomb_rate=0.2)
    elif name == "CbAS" or name == "DbAS":
        g = tf.keras.models.load_model("../notebooks/vae_initial_weights.h5")
        explorer = exp_fn(batch_size=100, virtual_screen=20, generator=g)
    else:
        explorer = exp_fn(batch_size=100, virtual_screen=20)
    explorer.debug = False

    save_path = f"../simulations/eval/RNA/{name}-GP/"
    os.makedirs(save_path, exist_ok=True)
    evaluator = Evaluator(
        explorer,
        landscape_types=LANDSCAPE_TYPES_RNA,
        path=save_path,
        adaptive_ensemble=False,
    )
    evaluator.evaluate_for_landscapes(evaluator.consistency_robustness_independence, num_starts=5)

# For TF
for name, exp_fn in pairs:
    if name == "AdaLead":
        explorer = exp_fn(batch_size=100, virtual_screen=20, recomb_rate=0.2)
    elif name == "CbAS" or name == "DbAS":
        g = tf.keras.models.load_model("../notebooks/vae_initial_weights.h5")
        explorer = exp_fn(batch_size=100, virtual_screen=20, generator=g)
    else:
        explorer = exp_fn(batch_size=100, virtual_screen=20)
    explorer.debug = False

    save_path = f"../simulations/eval/TF/{name}-GP/"
    os.makedirs(save_path, exist_ok=True)
    evaluator = Evaluator(
        explorer,
        landscape_types=LANDSCAPE_TYPES_TF,
        path=save_path,
        adaptive_ensemble=False,
    )
    evaluator.evaluate_for_landscapes(evaluator.consistency_robustness_independence, num_starts=13)