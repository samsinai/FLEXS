"""Utility functions for manipulating sequences."""
import random
from typing import List, Union

import numpy as np

AAS = "ILVAGMFYWEDQNHCRKSTP"
"""str: Amino acid alphabet for proteins (length 20 - no stop codon)."""

RNAA = "UGCA"
"""str: RNA alphabet (4 base pairs)."""

DNAA = "TGCA"
"""str: DNA alphabet (4 base pairs)."""

BA = "01"
"""str: Binary alphabet '01'."""


def construct_mutant_from_sample(
    pwm_sample: np.ndarray, one_hot_base: np.ndarray
) -> np.ndarray:
    """Return one hot mutant, a utility function for some explorers."""
    one_hot = np.zeros(one_hot_base.shape)
    one_hot += one_hot_base
    i, j = np.nonzero(pwm_sample)  # this can be problematic for non-positive fitnesses
    one_hot[i, :] = 0
    one_hot[i, j] = 1
    return one_hot


def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:
    """
    Return the one-hot representation of a sequence string according to an alphabet.

    Args:
        sequence: Sequence string to convert to one_hot representation.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        One-hot numpy array of shape `(len(sequence), len(alphabet))`.

    """
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out


def one_hot_to_string(
    one_hot: Union[List[List[int]], np.ndarray], alphabet: str
) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.

    Args:
        one_hot: One-hot of shape `(len(sequence), len(alphabet)` representing
            a sequence.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        Sequence string representation of `one_hot`.

    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])


def generate_single_mutants(wt: str, alphabet: str) -> List[str]:
    """Generate all single mutants of `wt`."""
    sequences = [wt]
    for i in range(len(wt)):
        tmp = list(wt)
        for j in range(len(alphabet)):
            tmp[i] = alphabet[j]
            sequences.append("".join(tmp))
    return sequences


def generate_random_sequences(length: int, number: int, alphabet: str) -> List[str]:
    """Generate random sequences of particular length."""
    return [
        "".join([random.choice(alphabet) for _ in range(length)]) for _ in range(number)
    ]


def generate_random_mutant(sequence: str, mu: float, alphabet: str) -> str:
    """
    Generate a mutant of `sequence` where each residue mutates with probability `mu`.

    So the expected value of the total number of mutations is `len(sequence) * mu`.

    Args:
        sequence: Sequence that will be mutated from.
        mu: Probability of mutation per residue.
        alphabet: Alphabet string.

    Returns:
        Mutant sequence string.

    """
    mutant = []
    for s in sequence:
        if random.random() < mu:
            mutant.append(random.choice(alphabet))
        else:
            mutant.append(s)
    return "".join(mutant)
