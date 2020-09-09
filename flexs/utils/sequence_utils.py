import random

import numpy as np
from scipy.stats import entropy

AAS = "ILVAGMFYWEDQNHCRKSTP"  # protein alphabet
RNAA = "UGCA"  # RNA alphabet
DNAA = "TGCA"  # DNA alphabet
BA = "01"  #  binary alphabet


def renormalize_moves(one_hot_input, rewards_output):
    """Ensures that staying in place gives no reward."""
    zero_current_state = (one_hot_input - 1) * (-1)
    return np.multiply(rewards_output, zero_current_state)


def make_random_action(shape):
    action = np.zeros(shape)
    action[np.random.randint(shape[0]), np.random.randint(shape[1])] = 1
    return action


def sample_greedy(matrix):
    i, j = matrix.shape
    max_arg = np.argmax(matrix)
    y = max_arg % j
    x = int(max_arg / j)
    output = np.zeros((i, j))
    output[x][y] = matrix[x][y]
    return output


def sample_random(matrix):
    i, j = matrix.shape
    non_zero_moves = np.nonzero(matrix)
    k = len(non_zero_moves)
    l = len(non_zero_moves[0])
    if k != 0 and l != 0:
        rand_arg = random.choice(
            [[non_zero_moves[alph][pos] for alph in range(k)] for pos in range(l)]
        )
    else:
        rand_arg = [random.randint(0, i - 1), random.randint(0, j - 1)]
    y = rand_arg[1]
    x = rand_arg[0]
    output = np.zeros((i, j))
    output[x][y] = matrix[x][y]
    return output


def construct_mutant_from_sample(pwm_sample, one_hot_base):
    one_hot = np.zeros(one_hot_base.shape)
    one_hot += one_hot_base
    i, j = np.nonzero(pwm_sample)  # this can be problematic for non-positive fitnesses
    one_hot[i, :] = 0
    one_hot[i, j] = 1
    return one_hot


def string_to_one_hot(sequence, alphabet):
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out


def translate_one_hot_to_string(one_hot, order_list):
    out = []
    for i in range(one_hot.shape[1]):
        ix = np.argmax(one_hot[:, i])
        out.append(order_list[ix])
    return "".join(out)


def break_down_sequence_to_singles(sequence_mask):
    """Currently only handles substitutions; can handle full sequences (not just masks)."""
    singles = []
    tmp = ["_"] * len(sequence_mask)
    for i in range(len(sequence_mask)):
        if sequence_mask[i] != "_":
            tmp[i] = sequence_mask[i]
            singles.append("".join(tmp))
            tmp = ["_"] * len(sequence_mask)
    return singles


def translate_mask_to_full_seq(sequence_mask, wt_seq):
    full_sequence = [""] * len(sequence_mask)
    count = 0
    for i, j in zip(sequence_mask, wt_seq):
        if i == "_":
            full_sequence[count] = wt_seq[count]
        else:
            full_sequence[count] = sequence_mask[count]
        count += 1
    return "".join(full_sequence)


def translate_full_seq_to_mask(full_sequence, wt_seq):
    mask_sequence = [""] * len(full_sequence)
    count = 0
    for i, j in zip(full_sequence, wt_seq):
        if i == j:
            mask_sequence[count] = "_"
        else:
            mask_sequence[count] = full_sequence[count]
        count += 1
    return "".join(mask_sequence)


def generate_single_mutants(wt, alphabet):
    sequences = [wt]
    for i in range(len(wt)):
        tmp = list(wt)
        for j in range(len(alphabet)):
            tmp[i] = alphabet[j]
            sequences.append("".join(tmp))
    return sequences


def generate_singles(length, function, wt=None, alphabet=AAS):
    """Generate a library of random single mutant fitnesses to use for hypothetical models."""
    singles = {}
    for i in range(length):
        tmp = ["_"] * length
        if wt:
            tmp = list(wt[:])
        for j in range(len(alphabet)):
            tmp[i] = alphabet[j]
            singles["".join(tmp)] = function()
            if wt:  # if wt is passed, construct such that additive fitness of wt is 0
                if alphabet[j] == wt[i]:
                    singles["".join(tmp)] = 0
    return singles


def generate_random_sequences(length, number, alphabet=AAS):
    """Generates random sequences of particular length."""
    sequences = []
    for i in range(number):
        tmp = []
        for j in range(length):
            tmp.append(random.choice(alphabet))
        sequences.append("".join(tmp))

    return sequences


def generate_random_mutant(sequence, mu, alphabet=AAS):
    mutant = []
    for s in sequence:
        if random.random() < mu:
            mutant.append(random.choice(alphabet))
        else:
            mutant.append(s)
    return "".join(mutant)


def generate_all_binary_combos(K):
    variants = [["0"], ["1"]]
    for _ in range(K):
        variants = expand_tree(variants)
    return variants


def expand_tree(list_of_nodes):
    expanded_tree = []
    for node in list_of_nodes:
        expanded_tree.extend([node + ["0"], node + ["1"]])
    return expanded_tree
