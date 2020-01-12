
import random 
from scipy.stats import entropy
import numpy as np

AAS="ILVAGMFYWEDQNHCRKSTP" # protein alphabet
RNAA="UGCA" # RNA alphabet
DNAA= "TGCA" # DNA alphabet
BA="01" #  binary alphabet

def translate_string_to_one_hot(sequence,order_list):
    out=np.zeros((len(order_list),len(sequence)))
    for i in range(len(sequence)):
        out[order_list.index(sequence[i])][i]=1
    return out

def translate_one_hot_to_string(one_hot,order_list):
    out=[]
    for i in range(one_hot.shape[1]):
        ix=np.argmax(one_hot[:,i])
        out.append(order_list[ix])
    return "".join(out)


def translate_aa_to_index(aa):
    return AAS.index(aa.upper())


def translate_index_to_aa(i):
    return AAS[i]

def break_down_sequence_to_singles(sequence_mask):
        """Currently only handles substitutions, can handle full sequences(not just masks)"""
        singles=[]
        tmp=["_"]*len(sequence_mask)
        for i in range(len(sequence_mask)):
            if sequence_mask[i]!="_":
               tmp[i]=sequence_mask[i]
               singles.append("".join(tmp))
               tmp=["_"]*len(sequence_mask)
        return singles


def translate_mask_to_full_seq(sequence_mask,wt_seq):
    full_sequence=[""]*len(sequence_mask)
    count=0
    for i,j in zip(sequence_mask,wt_seq):
        if i=="_":
           full_sequence[count]=wt_seq[count]
        else:
           full_sequence[count]=sequence_mask[count]
        count+=1
    return "".join(full_sequence)


def translate_full_seq_to_mask(full_sequence,wt_seq):
    mask_sequence=[""]*len(full_sequence)
    count=0
    for i,j in zip(full_sequence,wt_seq):
        if i==j:
           mask_sequence[count]="_"
        else:
           mask_sequence[count]=full_sequence[count]
        count+=1
    return "".join(mask_sequence)


def generate_single_mutants(wt,alphabet):
        sequences=[wt]
        for i in range(len(wt)):
            tmp=list(wt)
            for j in range(len(alphabet)):
                tmp[i]=alphabet[j]
                sequences.append("".join(tmp)) 
        return sequences

def generate_singles(length,function,wt=None,alphabet=AAS):
    """generate a library of random single mutant fitnesses to use for hypothetical models"""
    singles={}
    for i in range(length):
        tmp=["_"]*length
        if wt:
            tmp=list(wt[:])
        for j in range(len(alphabet)):
            tmp[i]=alphabet[j]
            singles["".join(tmp)]=function()
            if wt: #if wt is passed, construct such that additive fitness of wt is 0
                if alphabet[j]==wt[i]:
                   singles["".join(tmp)]=0
    return singles

def generate_random_sequences(length,number,alphabet=AAS):
    """generates random sequences of particular lenght"""
    sequences=[]
    for i in range(number):
        tmp=[]
        for j in range(length):
            tmp.append(random.choice(alphabet))
        sequences.append("".join(tmp))

    return sequences

def generate_random_mutant(sequence,mu,alphabet=AAS):
    mutant=[]
    for s in sequence:
        if random.random()<mu:
           mutant.append(random.choice(alphabet))
        else:
           mutant.append(s)
    return "".join(mutant) 


def generate_all_binary_combos(K):
    variants=[["0"],["1"]]
    for i in range(K): 
        variants=expand_tree(variants)
    return variants


def expand_tree(list_of_nodes):
    expanded_tree=[]
    for node in list_of_nodes:
        expanded_tree.extend([node+["0"],node+["1"]])
    return expanded_tree


def get_set_column_entropy(sequences,alphabet):
    seqs_int_array=[]
    for seq in sequences:
        row=[i for i in list(seq)]
        seqs_int_array.append(row)
    count_matrix=np.zeros((len(alphabet),len(seq)))
    seqs_int_array=np.array(seqs_int_array)
    for j in range(len(seq)):
        for i in range(len(alphabet)):
            count_matrix[i][j]=list(seqs_int_array[:,j]).count(alphabet[i])
    
    return entropy(count_matrix,base=2)


