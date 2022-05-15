import os
from typing import Dict

import numpy as np
import pandas as pd

import flexs
from flexs.types import SEQUENCES_TYPE


class GB1IgGBinding(flexs.Landscape):
    """
    A landscape of fitnesses based on stability binding affinity of the 4-AA GB1 domain of Protein G to IgG-FC.

    We use experimental data from Wu et al. which can be downloaded in its original form here:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4985287/
    """

    gb1_domain_wt_sequence = "VDGV"

    def __init__(self, landscape_fpath: str):
        super().__init__(name="GB1_IgG_Binding")
        data = pd.read_csv(landscape_fpath)
        self.sequences = dict(zip(data["sequence"], data["fitness"]))

    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        return np.array([self.sequences[seq] for seq in sequences])


def registry() -> Dict[str, Dict]:
    """
    Return a dictionary of problems of the form:

    ```python
    {
        "problem name": {
            "params": ...,
        },
        ...
    }
    ```

    where `flexs.landscapes.GB1IgGBinding(**problem["params"])` instantiates the
    GB1 landscape for the given set of parameters.

    Returns:
        Problems in the registry.

    """
    gb1_landscape_fpath = os.path.join(
        os.path.dirname(__file__), "data/gb1_igg_binding", "full_landscape.csv"
    )
    return {
        "GB1_IgG_Binding": {
            "params": {"landscape_fpath": gb1_landscape_fpath},
            "starts": [
                GB1IgGBinding.gb1_domain_wt_sequence,
            ],
        }
    }
