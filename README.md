# FLEXS: Fitness Landscape Exploration Sandbox

FLEXS is an open simulation environment that enables you to develop and compare model-guided biological sequence design algorithms. 

- [Installation](#installation)
- [Overview](#overview)
	- [Quickstart](Tutorial.ipynb) 
- [Major components](#major-components)

Your PR and contributions to this sandbox are most welcome. If you make use of it, please ensure that you cite the relevant original articles upon which this work was made possible (we provide links in this readme). 

## Installation

We strongly recommend that you install the dependencies for the sandbox in a conda virtual environment. 
The dependencies of the sandbox are the latest versions of the following packages:

* Numpy 
* Scikit-Learn
* Pandas 
* ViennaRNA 
* Tensorflow + Keras 
* Editdistance 

A minimal set of requirements can be installed with the following:
```
conda install pip -y
conda install -c bioconda viennarna -y
conda install scikit-learn -y
conda install pandas -y
pip install --upgrade tensorflow 
pip install keras 

```

If you are in a conda environment, you can also `./load_environment.sh` in the main directory to install all of the dependencies. The following dependencies are required for specific use case. 

* TQDM (If you plan on using the Evolutionary Bayesian Explorer)
* Tensorflow-Probability (If you aim to use DyNAPPO)
* TF-Agents  (If you aim to use DyNAPPO)
* TAPE (If you want to use the GFP oracle)
* Gin 

We also provide limited support for making landscapes with [Rosetta](https://www.rosettacommons.org/), note that it requires a separate license.  

## Overview

Biological sequence design through machine-guided directed evolution has been of increasing interest. This process often involves two closely connected steps:
  * Models `f` that attempt to learn the ground truth sequence `x` to function `y` relationships `g(x) = y`. 
  * Algorithms that explore the sequence space with the help of the trained model `f`. 

 
 While in some cases, these two steps are learned simultaneously, it is fairly common to have access to a well-trained model `f` which is *not* invertible. Namely, given a sequence `x`, the model can estimate `y'` (with variable accuracy), but it cannot generate a sequence `x'` associated with a specific function `y`. Therefore it is valuable to develop exploration algorithms `E(f)` that make use of the model `f` to propose sequences `x'`. 

 We implement a simulation environment that allows you to develop or port landscape exploration algorithms for a variety of challenging tasks. Our environment allows you to abstract away the model `f = Noisy_abstract_model(g)` or employ empirical models (like Keras/Pytorch or Sklearn models). You can see how these work in the [quickstart tutorial](Tutorial.ipynb). 

Our abstraction is comprised of three levels:
#### 1.  Ground truth oracles (landscapes) 
These oracles `g` are simulators that are assumed as ground truth, i.e. when queried, they return the true value `y_i` associated with a sequence `x_i`. Currently we have four classes of ground truth oracles implemented. 
- *[Transcription factor binding data](#transcription-factor-binding)*. This is comprised of 158 (experimentally) fully characterized landscapes. 
- *[RNA landscapes](#rna-landscapes)*. A set of curated and increasingly challenging RNA binding landscapes as simulated with ViennaRNA. 
- *[AAV Additive Tropism](#additive-aav-tropism)*. A hypothesized noisy additive protein landscape based on tissue tropism of single mutant AAV2 capsid protein.   
- *[GFP fluorescence](#gfp-fluorescence)*. Fluorescence of GFP protein as predicted by TAPE transformer model. 

For all landscapes we also provide a fixed set of initial points with different degrees of previous optimization, so that the relative strength of algorithms when starting from locations near or far away from peaks can be evaluated. 

#### 2. Noisy oracles
We have two types of noisy oracles `f`. 
- **Noisy_abstract_model**: These models get access to the ground truth `g`, but do not allow the explorer to access `g` directly. They corrupt the signal from `g` but adding noise to it, proportional to the distance of the query from the (nearest) observed data. The parameter `signal_strength` which is between 0 (no signal) and 1 (perfect model) determines the rate of decay.  
- **Empirical models**: These models train a standard algorithm on the observed data. The currently available architectures can be found in `utils/model_architectures`. 
All noisy models can be ensembled using ensemble class. Ensembles also have the ability to be *adaptive* i.e. the models within an ensemble will be reweighted based on their accuracy on the last measured set. 

#### 3. Exploration algorithms

 Exploration algorithms have access to `f` with some limit on the number of queries to this oracle `virtual_screen`. Once they have queried that many samples, they would commit to measuring `batch_size` from the ground truth, which incurrs a real cost. The class `base_explorer` implements the housekeeping tasks, and new exploration algorithms can be implemented by inheriting from it.  

#### 4. Evaluators




# Major components

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
Predicting RNA secondary structures is a well-studied problem. There are efficient and accurate dynamic programming approaches to calculates secondary structure of short RNA sequences. These landscapes give us a good proxy for a consistent oracle over entire domain of large landscapes.  We use the [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) package to simulate binding landscapes of RNA sequences as a ground truth oracle.

Our sandbox allows for constructing arbitrarily complex landscapes (although we discourage large RNA sequences as the accuracy of the simulator deteriorates above 200 nucleotides). As benchmark, we provide a series of 36 increasingly complex RNA binding landscapes. These landscapes each come with at least 5 suggested starting sequences, with various initial optimization. 

The simplest landscapes are binding landscapes with a single hidden target (often larger than the design sequence resulting in multiple peaks). The designed sequences is meant to be optimized to bind the target with the minimum binding energy (we use duplex energy as our objective). We estimate `optimal(y)` by computing the binding energy of the perfect complement of the target and normalize the fitnesses using that measure (hence this is only an approximation and often a slight underestimate). RNA landscapes show many local peaks, and often multiple global peaks due to symmetry. 

Additionally, we construct more complex landscapes by increasing the number of hidden targets, enforcing specific conservation patterns, and composing the scores of each landscapes multiplicatively. See `utils/multi_dimensional_model.py` for the generic class that allows composing landscapes.  


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

 Ogden et al. (2019) perform a comprehensive single mutation scan of AAV2 capsid protein, assaying tropism for five different target tissues. The authors show that an additive model is informative about the local structure of the landscape. Here we use the data from the single mutations to generate a toy additive model. Here `y' := sum(s_i)+ e`, where `i` indicates the position across the sequences, and `s_i` indicates a sequence with mutation `s` at position `i` and `e` indicates iid Gaussian noise. This construct is also known as "Rough Mt. Fuji" (RMF) and many empirical fitness landscapes are consistent with an RMF local structure around viable natural sequences with unpredictable regions in between. In the noise-free setting, the RMF landscape is convex with a single peak. We allow the construction of multiple target tissues, and different design lengths (tasks ranging from desiging short region of the protein to tasks that encompass designing the full protein). The scores are normalized between `[0,1]`. 

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
 In [TAPE](https://github.com/songlab-cal/tape), the authors benchmark multiple machine learning methods on a set of tasks including GFP fluorescence prediction. The GFP task is comprised of training and predicting fluorescence values on approximately 52,000 protein sequences of length 238 which are derived from the naturally occurring GFP in *Aequorea victoria* (See [this paper](https://www.nature.com/articles/nature17995)). Downloading and doing inference with this model is memory and time intensive. These landscapes are not normalized and therefore scores higher than 1 are possible (we do not know the maximum activation for the model). 

```
@inproceedings{tape2019,
author = {Rao, Roshan and Bhattacharya, Nicholas and Thomas, Neil and Duan, Yan and Chen, Xi and Canny, John and Abbeel, Pieter and Song, Yun S},
title = {Evaluating Protein Transfer Learning with TAPE},
booktitle = {Advances in Neural Information Processing Systems}
year = {2019}
}

@article{sarkisyan2016local,
  title={Local fitness landscape of the green fluorescent protein},
  author={Sarkisyan, Karen S and Bolotin, Dmitry A and Meer, Margarita V and Usmanova, Dinara R and Mishin, Alexander S and Sharonov, George V and Ivankov, Dmitry N and Bozhanova, Nina G and Baranov, Mikhail S and Soylemez, Onuralp and others},
  journal={Nature},
  volume={533},
  number={7603},
  pages={397--401},
  year={2016},
  publisher={Nature Publishing Group}
}
```
### Noisy Oracles

#### Noisy Abstract Models

#### Empirical Models

### Exploration Algorithms
	[Base Explorer](explorers/base_explorer.py)
	[Random Explorer](explorers/random_explorer.py)
#### Evolutionary Algorithms
	 [Wright-Fisher, Model-guided Wright Fisher](explorers/evolutionary_explorers.py)
	 [CMA-ES](explorers/CMAES_explorer.py)
	 [Independent sites X-entropy, ADALEAD](explorers/elitist_explorers.py)
#### DbAS and CbAS
	 Adaptation of [CbAS and DbAS](explorers/CbAS_DbAS_explorers.py)
#### Reinforcement Learning Algorithms
     [DQN]()
     [PPO]()
     [DyNAPPO]()		
#### Bayesian Optimization 
	 [Evolutionary BO]()
	 [Enumerative BO]()



 

