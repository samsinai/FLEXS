![FLEXS Logo](LOGO.png)

[![build status](https://github.com/samsinai/FLEXS/workflows/build/badge.svg)](https://github.com/samsinai/FLEXS/actions) [![Documentation Status](https://readthedocs.org/projects/flexs/badge/?version=latest)](https://flexs.readthedocs.io/en/latest/?badge=latest) [![PyPI package](https://img.shields.io/pypi/v/flexs)](https://pypi.org/project/flexs/)

💪 FLEXS is an open-source simulation environment that enables you to develop and compare model-guided biological sequence design algorithms. This project was developed with support from [Dyno Therapeutics](https://www.dynotx.com).

- [Installation](#installation)
- [Overview](#overview)
- [Tutorial](examples/Tutorial.ipynb)
- [Contribution and credits](#contributions-and-credits) 
- [Components](#components)
  - [Ground truth landscapes](#ground-truth-landscapes)
  - [Noisy oracles](#noisy-oracles)
  - [Exploration algorithms](#exploration-algorithms)
     - [Bring your own explorer](#bring-your-own-explorer) 



## Installation

FLEXS is available on [PyPI](https://pypi.org/project/flexs/) 🐍 and can be installed with `pip install flexs`.

There are two optional, but very useful dependencies, [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) (for RNA binding landscapes) and [PyRosetta](http://www.pyrosetta.org) (for protein design landscapes). These can both be installed with conda:
```bash
$ conda install -c bioconda viennarna
$ conda install pyrosetta  # Set up RosettaCommons conda channel first (http://www.pyrosetta.org/dow)
```
Note that PyRosetta requires a commercial license if not being used for academic purposes. 

IMPORTANT: ViennaRNA seems to have issues with Python 3.8, so try to run in a Python 3.5>=, <=3.7 environment.

If contributing or running paper code/experiments, we recommend that you install the dependencies for the sandbox in a conda virtual environment. You can initialize a new Python 3.7 environement with `conda env create --name {env_name} python=3.7`. Then install the local version of `flexs` with `pip install -e .` in the root directory.


## Overview

Biological sequence design through machine-guided directed evolution has been of increasing interest. This process often involves two closely connected steps:
  * Models `f` that attempt to learn the ground truth sequence to function relationship `g(x) = y`. 
  * Algorithms that explore the sequence space with the help of the trained model `f`. 


 While in some cases, these two steps are learned simultaneously, it is fairly common to have access to a well-trained model `f` which is *not* invertible. Namely, given a sequence `x`, the model can estimate `y'` (with variable accuracy), but it cannot generate a sequence `x'` associated with a specific function `y`. Therefore it is valuable to develop exploration algorithms `E(f)` that make use of the model `f` to propose sequences `x'`. 

 We implement a simulation environment that allows you to develop or port landscape exploration algorithms for a variety of challenging tasks. Our environment allows you to abstract away the model `f = Noisy_abstract_model(g)` or employ empirical models (like Keras/Pytorch or Sklearn models). You can see how these work in the **[tutorial](examples/Tutorial.ipynb)**. 

Our abstraction is comprised of four levels:
#### 1.  Fitness Landscapes 🏔️
These oracles `g` are simulators that are assumed as ground truth, i.e. when queried, they return the true value `y_i` associated with a sequence `x_i`. Currently we have four classes of ground truth oracles implemented. 
- *[Transcription factor binding data](#transcription-factor-binding)*. This is comprised of 158 (experimentally) fully characterized landscapes. 
- *[RNA landscapes](#rna-landscapes)*. A set of curated and increasingly challenging RNA binding landscapes as simulated with ViennaRNA. 
- *[AAV Additive Tropism](#additive-aav-tropism)*. A hypothesized noisy additive protein landscape based on tissue tropism of single mutant AAV2 capsid protein.   
- *[GFP fluorescence](#gfp-fluorescence)*. Fluorescence of GFP protein as predicted by TAPE transformer model. 
- *[Rosetta-based design](#rosetta-based-design)*. Rosetta-based design task for 3MSI anti-freeze protein. 

For all landscapes we also provide a fixed set of initial points with different degrees of previous optimization, so that the relative strength of algorithms when starting from locations near or far away from peaks can be evaluated. 

#### 2. Noisy oracles
 
Noisy oracles are (approximate) models `f` of the original ground truth landscape `g`. These allow for the exploration algorithm to screen sequences virtually, before committing to making expensive queries to `g`.  We implement two flavors of these
- Noisy abstract models: Noise corrupted version of `g` (this allows for independent study of exploration algorithms). 
- Empirical models: `f` is learned directly from the data that was collected so far. 

#### 3. Exploration algorithms 🕵️

 Exploration algorithms have access to `f` with some limit on the number of queries to this oracle `virtual_screen`. Once they have queried that many samples, they would commit to measuring `batch_size` from the ground truth, which incurrs a real cost. The class `base_explorer` implements the housekeeping tasks, and new exploration algorithms can be implemented by inheriting from it.  

#### 4. Evaluators 📊

We also implement a suite of [evaluation modules](flexs/evaluate.py) that automatically collect data that is necessary for evaluating algorithms on different performance criteria.

- *robustness*: Produces data for analyzing how explorer performance changes given different quality of models.
- *efficiency*: Produces data for analyzing how explorer performance changes when more computational evaluations are allowed.
- *adaptivity*: Produces data for analyzing how the explorer is sensitive to the number of batches it is allowed to sample, given a fixed total budget.

See the [tutorial](examples/Tutorial.ipynb) for an example of how these can be run. 

## Contributions and credits 🤩
Your PR and contributions to this sandbox are most welcome. If you make use of data or algorithms in this sandbox, please ensure that you cite the relevant original articles upon which this work was made possible (we provide links in this readme). To cite the sandbox itself:
```
@article{sinai2020adalead,
  title={AdaLead: A simple and robust adaptive greedy search algorithm for sequence design},
  author={Sinai, Sam and Wang, Richard and Whatley, Alexander and Slocum, Stewart and Locane, Elina and Kelsic, Eric},
  journal={arXiv preprint},
  year={2020}
}
```

FLEXS 0.2.1 was developed by Sam Sinai, Richard Wang, Alexander Whatley, Elina Locane, and Stewart Slocum.  

## Components

### Ground Truth Landscapes

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

#### RNA Landscapes
Predicting RNA secondary structures is a well-studied problem. There are efficient and accurate dynamic programming approaches to calculates secondary structure of short RNA sequences. These landscapes give us a good proxy for a consistent oracle over entire domain of large landscapes.  We use the [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) package to simulate binding landscapes of RNA sequences as a ground truth oracle.

Our sandbox allows for constructing arbitrarily complex landscapes (although we discourage large RNA sequences as the accuracy of the simulator deteriorates above 200 nucleotides). As benchmark, we provide a series of 36 increasingly complex RNA binding landscapes. These landscapes each come with at least 5 suggested starting sequences, with various initial optimization. 

The simplest landscapes are binding landscapes with a single hidden target (often larger than the design sequence resulting in multiple peaks). The designed sequences is meant to be optimized to bind the target with the minimum binding energy (we use duplex energy as our objective). We estimate `optimal(y)` by computing the binding energy of the perfect complement of the target and normalize the fitnesses using that measure (hence this is only an approximation and often a slight underestimate). RNA landscapes show many local peaks, and often multiple global peaks due to symmetry. 

Additionally, we construct more complex landscapes by increasing the number of hidden targets, enforcing specific conservation patterns, and composing the scores of each landscapes multiplicatively. See [multi-dimensional models](flexs/ensemble.py) for the generic class that allows composing landscapes.  


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

#### Additive AAV landscapes

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

#### GFP 
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
#### Rosetta-based Design 

[Rosetta](https://www.rosettacommons.org/software) is a protein modeling software suite used for *de novo* design and structure prediction. Based on the principle that structure determines function, the Rosetta design process begins with a desired 3-D protein conformation and tries to find amino acid sequences that are likely to fold to that structure. While the dynamics of protein folding are still poorly understood, this approach has proven remarkably effective in practice, and so we find it an acceptable analogue to the true fitness landscape. To keep our experiments computationally feasible, we omit the expensive step of side-chain packing and use the simplified centroid scoring frounction as our objective. We use the [PyRosetta](http://www.pyrosetta.org) Python interface to Rosetta. The Rosetta design objective function is a scaled estimate of the folding energy, which has been found to be an indicator of the probability that a sequence will fold to the desired structure. As an example, we provide an optimization challenge for the structure of 3MSI, a 66 amino acid antifreeze protein found in the ocean pout starting from 5 sequences with 3-27 mutations from the wildtype. Here, we normalize energy scores by scaling and shifting their distribution and then applying the sigmoid function.


```
@article{chaudhury2010pyrosetta,
  title={PyRosetta: a script-based interface for implementing molecular modeling algorithms using Rosetta},
  author={Chaudhury, Sidhartha and Lyskov, Sergey and Gray, Jeffrey J},
  journal={Bioinformatics},
  volume={26},
  number={5},
  pages={689--691},
  year={2010},
  publisher={Oxford University Press}
}
```


### Noisy Oracles

#### Noisy Abstract Models
These models get access to the ground truth `g`, but do not allow the explorer to access `g` directly. They corrupt the signal from `g` but adding noise to it, proportional to the distance of the query from the (nearest) observed data. The parameter `signal_strength` which is between 0 (no signal) and 1 (perfect model) determines the rate of decay.  


#### Empirical Models
These models train a standard algorithm on the observed data. Some baseline models can be found in [flexs/baselines/models](flexs/baselines/models). All landscapes and models can also be ensembled using the [ensemble class](flexs/ensemble.py). Ensembles also have the ability to be *adaptive* i.e. the models within an ensemble will be reweighted based on their accuracy on the last measured set.


### Exploration Algorithms

#### Bring your own explorer
Exploration algorithms are search methods that use noisy oracles to select the next batch of samples from the landscape. This is the main service of this sandbox, you can implement your own explorer by simply inheriting from the [Base Explorer](flexs/explorer.py), and implementing a single method:


```python
class MyExplorer(flexs.Explorer):
    """Your explorer here"""
    def __init__(self,
                 model,
                 rounds,
                 starting_sequence,
                 sequences_batch_size,
                 model_queries_per_batch,
                 **kwargs):

        name = f"MyExplorer_{**kwargs}"
        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
        )
        "Your custom attributes here"

    def propose_sequences(self, measured_sequences_data):
        """
        Your method implementation overriding the main explorer.
        It is allowed to make *model_queries_per_batch* queries to the model
        and make *sequences_batch_size* proposals in return.
        """

        return sequences, scores
```

#### Baseline Explorers

- [Random Explorer](flexs/baselines/explorers/random.py): A baseline random explorer.

#### Evolutionary Algorithms
- [Naive Genetic Algorithm, Wright-Fisher](flexs/baselines/explorers/genetic_algorithm.py): A standard Wright-Fisher process that has access to an oracle for pre-screening. 

- [CMA-ES](flexs/baselines/explorers/cmaes.py): The CMA-ES algorithm that optimizes a continuous relaxation of one-hot vectors encoding sequences (another evolutionary baseline). 

- [ADALEAD](flexs/baselines/explorers/adalead.py) ⭐️: ADALEAD is our recommended "benchmark" algorithm as it is robust to hyperparameters, and is relatively fast in execution. It also compares strongly to other state of the art algorithms.

```
@article{sinai2020adalead,
  title={AdaLead: A simple and robust adaptive greedy search algorithm for sequence design},
  author={Sinai, Sam and Wang, Richard and Whatley, Alexander and Slocum, Stewart and Locane, Elina and Kelsic, Eric},
  journal={arXiv preprint},
  year={2020}
}
```

#### DbAS and CbAS
- [CbAS and DbAS](flexs/baselines/explorers/explorers/cbas_dbas.py)
```
@article{brookes2019conditioning,
  title={Conditioning by adaptive sampling for robust design},
  author={Brookes, David H and Park, Hahnbeom and Listgarten, Jennifer},
  journal={arXiv preprint arXiv:1901.10060},
  year={2019}
}
@article{brookes2018design,
  title={Design by adaptive sampling},
  author={Brookes, David H and Listgarten, Jennifer},
  journal={arXiv preprint arXiv:1810.03714},
  year={2018}
}
```

#### Reinforcement Learning Algorithms
Adaptations of the following RL algorithms.

- [DQN](flexs/baselines/explorers/dqn_explorer.py)

- [PPO](flexs/baselines/explorers/ppo.py)

- [DyNAPPO](flexs/baselines/explorers/dyna_ppo.py): See the following citation.
```
@inproceedings{angermueller2019model,
  title={Model-based reinforcement learning for biological sequence design},
  author={Angermueller, Christof and Dohan, David and Belanger, David and Deshpande, Ramya and Murphy, Kevin and Colwell, Lucy},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```	
#### Bayesian Optimization 

- [Evolutionary/Enumerative BO](flexs/baselines/explorers/bo.py): Bayesian optimization with sparse sampling of the mutation space. A fully enumerated (when possible) is also implemented mutation space.
