
# Topological Phase Calssification of Water

This package (Top2Phase) implements the graph neural network for classificaiton of water phases, it constructes graphs based on the positioanal information of neighboring water molecules obtained from molecular dynamics trajectories and train graph neural network model to classify these phase using only edge information.

For further information see following paper.
[Top2Phase](https://doi.org)

![](images/image.png)

## Table of Contents

- [How to cite](#how-to-cite)
- [Installation](#Installation)
- [Usage](#usage)
  - [Trajectory to Graph](#MD-Data-Processing)
  - [Training](#Graph-neural-network-training)
  - [Analysis using gradient](#Saliency-Map)
  - [Analysis using masking](#masking-Explaination)
  - [Visualize the results](#visualize-the-results)
- [Data](#data)
- [Authors](#authors)
- [License](#license)

Topological Classification of Water Phases Using Edge-Conditioned Convolutional Graph Neural Network  


Python codes are located in the src/. directory:


1. compute_orderparameters.py: calculate order parameters from md trajectory
2. compute_pairwisedistance.py: computes pairwise distance for all atoms within a cut-off distance 
3. utility.py: codes to support other python scriipts for reading and managing dataset including graphs for graph neural network and md trajectory for preprocessing
4. model.py: graph neural network model functional API
5. Top2Phase.py: neural network model for classificaiton is implemented. 
6. main.py : training of the network model for classificaiton and model evaluation 

## Installation
  We recommend installaiton of following packages with this code, preferabely all in a conda enviroment
  ```
  conda create -n top2phase python=3.6 tensorflow pymatgen tqdm -c anaconda -c conda-forge
  source activate top2phase 
  pip install pyboo
  pip install spektral==1.0.4
  ```
  ### Requirment
   [tensorflow >= 2.1](https://www.tensorflow.org)
   [spektral >= 1.0.4](https://graphneural.network)
   [pyboo](https://pyboo.readthedocs.io/en/latest/index.html) 
   [pymatgen](http://pymatgen.org)
   [tqdm](https://tqdm.github.io)
## Usage

### Trajectory to Graph




### Graph Neural Network Training
```
python main.py -h
usage: ECCConv net training  for phase classification [-h]
                                                      [--list_of_graphs LIST_OF_GRAPHS [LIST_OF_GRAPHS ...]]
                                                      [--dataset_size DATASET_SIZE]
                                                      [--learning_rate LEARNING_RATE]
                                                      [--batch_size BATCH_SIZE]
                                                      [--epochs EPOCHS]
                                                      [--data_path DATA_PATH]
                                                      [--random_seed RANDOM_SEED]
                                                      [--split_fraction SPLIT_FRACTION]
                                                      [--savemodel SAVEMODEL]
                                                      [--save_freq SAVE_FREQ]
                                                      [--list_of_phases LIST_OF_PHASES [LIST_OF_PHASES ...]]

optional arguments:
  -h, --help            show this help message and exit
  --list_of_graphs LIST_OF_GRAPHS [LIST_OF_GRAPHS ...]
                        list of graphs to use for training save as npz, it
                        should be generated using the convert_to_graph.py
  --dataset_size DATASET_SIZE
                        dataset size (int)
  --learning_rate LEARNING_RATE
                        learning rate
  --batch_size BATCH_SIZE
                        batch size for each epochs
  --epochs EPOCHS       total number of training epochs
  --data_path DATA_PATH
                        path to the data file
  --random_seed RANDOM_SEED
                        random seed for consistent training
  --split_fraction SPLIT_FRACTION
                        fraction of data for validation
  --savemodel SAVEMODEL
                        name of check file to save the model
  --save_freq SAVE_FREQ
                        frequency to save the model
  --list_of_phases LIST_OF_PHASES [LIST_OF_PHASES ...]
                        map graph to phase, start from 0 to n-1
```

*Example*
```
python main.py --list_of_graphs graph_Icehvapor_low_high_24.npz graph_Icehvapor_low_high_30.npz graph_Icehvapor_low_high_21.npz --list_of_phases 0 1 0 --dataset_size 21000 --batch_size 64 --learning_rate 0.0001 --epochs 5000 --data_path <dir> --save_freq 200
```

