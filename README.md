# FLEXS: Fitness Landscape Exploration Sandbox (for model-guided biological sequence design)

FLEXS is a simulation environment that enables you to develop and compare model-guided biological sequence design algorithms.  

## Installation



## Overview

Biological sequence design through machine-guided directed evolution has been of increasing interest. This process often involves two closely connected steps:
  * Models `f` that attempt to learn the ground truth sequence `x` to function `y` relationships `g(x) = y`. 
  * Algorithms that explore the sequence space with the help of the trained model `f`. 

 
 While in some cases, these two steps are learned simultaneously, it is fairly common to have access to a well-trained model `f` which is *not* invertible. Namely, given a sequence `x`, the model can estimate `y'` (with variable accuracy), but it cannot generate a sequence `x'` associated with a specific function `y`. Therefore it is valuable to develop exploration algorithms `E(f)` that make use of the model `f` to propose sequences `x'`. 

 We implement a simulation environment that allows you to develop or port landscape exploration algorithms for a variety of challenging tasks. Our environment allows you to abstract away the model `f = Noisy_abstract_model(g)` or employ empirical models (like Keras/Pytorch or Sklearn models). You can see how these work in the tutorial. 

 ### Simulation

Our abstraction is comprised of three levels:
1.  **Ground truth oracles (landscapes)**: These oracles `g` are simulators that are assumed as ground truth, i.e. when queried, they return the true value `y_i` associated with a sequence `x_i`. Currently we have four classes of ground truth oracles implemented. 
	- *[Transcription factor binding data](#transcription-factor-binding)*. This is comprised of 158 (experimentally) fully characterized landscapes. 
	- *[RNA landscapes](#rna-landscapes)*. A set of curated and increasingly challenging RNA binding landscapes as simulated with ViennaRNA. 
	- *[AAV Additive Tropism](#additive-aav-tropism)*. A hypothesized noisy additive protein landscape based on tissue tropism of single mutant AAV2 capsid protein.   
	- *[GFP fluorescence](#gfp-fluorescence)*. Fluorescence of GFP protein as predicted by TAPE transformer model. 

For all landscapes we also provide a fixed set of initial points with different degrees of previous optimization, so that the relative strength of algorithms when starting from locations near or far away from peaks can be evaluated. 

2.  **Noisy oracles**: We have two types of noisy oracles `f`. 
	- Noisy_abstract_model: These models get access to the `g`, but do not allow the explorer to access `g` directly. They corrupt the signal from `g` but adding noise to it, proportional to the distance of the query from the observed data. The parameter `signal_strength` which is between 0 (no signal) and 1 (perfect model) determines the rate of decay.  
	- Empirical models: These models train a standard algorithm (selected from a suite, or new ones can be implements) on the data observed data. The currently available architectures can be found in `utils/model_architectures`. 
	All noisy models can be ensembled using ensemble class. Ensembles also have the ability to be *adaptive* i.e. the models within an ensemble will be reweighted based on their accuracy on the last measured set. 
3.  **Exploration algorithms**: This is where the experimentation happens. Exploration algorithms have access to `f` with some limit on the number of queries `virtual_screen`. Once they have queried that many samples, they would commit to measuring `batch_size` from the ground truth, which incurrs the "real" cost to the algorithm. 


# Further details

### Ground truth Landscapes

#### Transcription Factor Binding

Barrera et al. (2016) surveyed the binding affinity of more than one hundred and fifty transcription factors (TF) to all possible DNA sequences of length 8. Since the ground truth is entirely characterized, and biological, it is a relevant benchmark for our purpose. These generate the full picture for landscapes of size `4^8`. We shift the function distribution such that `y` is within `[0,1]`, and therefore `optimal(y)=1`. We also provide 15 initiation sequences with different degrees of optimization across landscapes. The sequence `TTAATTAA` for instance is a famous binding site that is a global peak in 20 of these landscapes, and a local peak (above all its single mutant neighbors) in 96 landscapes overall. `GCTCGAGC` is a local peak in 106 landscapes, whereas `AAAGAGAG` is not a peak in any of the 158 landscapes. It is notable that while complete, these landscapes are generally easy to optimize on due to their size. So we recommend that they are tested in very low-budget setting or additional classes of landscapes are used for benchmarking. 

```
@article{barrera2016survey,
  title={Survey of variation in human transcription factors reveals prevalent DNA binding changes},
  author={Barrera, Luis A and Vedenko, Anastasia and Kurland, Jesse V and Rogers, Julia M and Gisselbrecht, Stephen S and Rossin, Elizabeth J and Woodard, Jaie and Mariani, Luca and Kock, Kian Hong and Inukai, Sachi and others},
  journal={Science},
  volume={351},
  number={6280},
  pages={1450--1454},
  year={2016},
  publisher={American Association for the Advancement of Science}
}
```

### RNA Landscapes
Predicting RNA secondary structures is a well-studied problem. 
There are efficient and accurate dynamic programming approaches to calculates secondary structure of short RNA sequences. These landscapes give us a good proxy for a consistent oracle over entire domain of large landscapes.  We use the ViennaRNA package to simulate binding landscapes of RNA sequences as a ground truth oracle $\phi$ \cite{lorenz2011viennarna}. Predicting RNA secondary structures is a well-studied problem. 
There are efficient and accurate dynamic programming approaches to calculates secondary structure of short RNA sequences. These landscapes give us a good proxy for a consistent oracle over entire domain of large landscapes.  We use the ViennaRNA package to simulate binding landscapes of RNA sequences as a ground truth oracle $\phi$ \cite{lorenz2011viennarna}. 

Our sandbox allows for constructing arbitrarily complex landscapes (although we discourage large RNA sequences as the accuracy of the simulator deteriorates above 200 nucleotides). As benchmark, we provide a series of 36 increasingly complex RNA binding landscapes. These landscapes each come with at least 5 suggested starting sequences, with various fitness scores. 

The simplest landscapes are binding landscapes with a hidden target. The designed sequences is meant to be optimized to bind the target with the minimum binding energy (we use duplex energy as our objective). We estimate `optimal(y)` by computing the binding energy of the perfect complement of the target and normalize the fitnesses using that measure (hence this is only an approximation and often a slight underestimate). RNA landscapes show many local peaks, and often multiple global peaks due to symmetry. 

We further complicate the landscapes by combining binding energies to 2 hidden targets and calculating the score of as `sqrt(y_1 * y_2)`. Finally, we define composite "swampland" landscapes by setting $y$ sequences without a specific pattern to zero. This results in large areas of the landscape to show no gradient. .  

```
@article{lorenz2011viennarna,
  title={{ViennaRNA} Package 2.0},
  author={Lorenz, Ronny and Bernhart, Stephan H and Zu Siederdissen, Christian H{\"o}ner and Tafer, Hakim and Flamm, Christoph and Stadler, Peter F and Hofacker, Ivo L},
  journal={Algorithms for molecular biology},
  volume={6},
  number={1},
  pages={26},
  year={2011},
  publisher={Springer}
}
```

### Additive AAV landscapes

 Ogden et al. (2019) perform a comprehensive single mutation scan of AAV2 capsid protein, assaying tropism for five different target tissues. The authors show that an additive model is informative about the local structure of the landscape. Here we use the data from the single mutations to generate a toy additive model. Here $\phi'(\vec{x}):= \sum_i \phi(\vec{\sigma}^{(i)})+\eta$, where $i$ indicates the position across the sequences, and $\sigma^{(i)}$ indicates a sequence with mutation $\sigma$ at position $i$ and $\eta$ is Gaussian noise. This construct is also known as ``Rough Mt. Fuji" (RMF) and many empirical fitness landscapes are consistent with an RMF local structure around viable natural sequences with unpredictable regions in between. In the noise-free setting, the RMF landscape is convex with a single peak. 


```
@article{ogden2019comprehensive,
  title={Comprehensive AAV capsid fitness landscape reveals a viral gene and enables machine-guided design},
  author={Ogden, Pierce J and Kelsic, Eric D and Sinai, Sam and Church, George M},
  journal={Science},
  volume={366},
  number={6469},
  pages={1139--1143},
  year={2019},
  publisher={American Association for the Advancement of Science}
}
```

### GFP 

```
@inproceedings{tape2019,
author = {Rao, Roshan and Bhattacharya, Nicholas and Thomas, Neil and Duan, Yan and Chen, Xi and Canny, John and Abbeel, Pieter and Song, Yun S},
title = {Evaluating Protein Transfer Learning with TAPE},
booktitle = {Advances in Neural Information Processing Systems}
year = {2019}
}
```

