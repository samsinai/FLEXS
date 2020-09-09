import sys

sys.path.append("../")
import RNA
from flexs.evaluators.Evaluator import Evaluator
from flexs.baselines.explorers.bo_explorer import BO_Explorer

LANDSCAPE_TYPES_RNA = {"RNA": [0], "TF": []}
bo_explorer_prod = BO_Explorer(batch_size=10, virtual_screen=20)
bo_explorer_prod.debug = False
evaluator_bo = Evaluator(
    bo_explorer_prod,
    landscape_types=LANDSCAPE_TYPES_RNA,
    path="../simulations/eval/",
)
evaluator_bo.evaluate_for_landscapes(
    evaluator_bo.consistency_robustness_independence, num_starts=5
)
