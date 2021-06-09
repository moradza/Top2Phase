
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
  git clone https://github.com/moradza/Top2Phase.git
  cd Top2Phase
  pip3 install .
  ```
  ### Requirment
   [tensorflow >= 2.1](https://www.tensorflow.org)<br>
   [spektral >= 1.0.4](https://graphneural.network)<br>
   [pyboo](https://pyboo.readthedocs.io/en/latest/index.html) <br>
   [pymatgen](http://pymatgen.org)<br>
   [tqdm](https://tqdm.github.io)<br>
   [mdtraj](https://mdtraj.org/1.9.3/index.html)<br>
## Usage

### Trajectory to Graph

```
 pwdist -h
usage: The traj.xtc convertor to npz [-h] [--xtc_file XTC_FILE]
                                     [--gro_file GRO_FILE]
                                     [--npz_file NPZ_FILE] [--radius RADIUS]
                                     [--skip SKIP]

optional arguments:
  -h, --help           show this help message and exit
  --xtc_file XTC_FILE  path to the xtc file
  --gro_file GRO_FILE  path to gro file
  --npz_file NPZ_FILE  path to save pairwise distance as a npz file
  --radius RADIUS      radius for neigh list
  --skip SKIP          skip frames
```


```
 orderparms -h
usage: The traj.xtc convertor to order parameter npz [-h] [--xtc_file XTC_FILE]
                                     [--gro_file GRO_FILE]
                                     [--npz_file NPZ_FILE] [--radius RADIUS]
                                     [--skip SKIP]

optional arguments:
  -h, --help           show this help message and exit
  --xtc_file XTC_FILE  path to the xtc file
  --gro_file GRO_FILE  path to gro file
  --npz_file NPZ_FILE  path to save orderparameters as a npz file
  --radius RADIUS      radius for neigh list
  --skip SKIP          skip frames

```

*Example*
```
 for i in $(seq 1 31); do pwdist --xtc_file Iceh_vapor/temp_${i}_short.xtc  --gro_file Iceh_vapor/conf.gro --radius 0.60 --npz_file graph_Icehvapor_0.60_${i}.npz; done
 
  for i in $(seq 1 31); do orderparms --xtc_file Iceh_vapor/temp_${i}_short.xtc  --gro_file Iceh_vapor/conf.gro --radius 0.60 --npz_file OP_Icehvapor_0.60_${i}.npz; done
```




### Graph Neural Network Training
```
python main.py -h
usage: ECCConv net training for phase classification [-h]
                                                      [--list_of_graphs LIST_OF_GRAPHS [LIST_OF_GRAPHS ...]]
                                                      [--dataset_size DATASET_SIZE]
                                                      [--learning_rate LEARNING_RATE]
                                                      [--batch_size BATCH_SIZE]
                                                      [--epochs EPOCHS]
                                                      [--data_path DATA_PATH]
                                                      [--random_seed RANDOM_SEED]
                                                      [--split_fraction SPLIT_FRACTION]
                                                      [--save_model_dir SAVE_MODEL_DIR]
                                                      [--list_of_phases LIST_OF_PHASES [LIST_OF_PHASES ...]]
                                                      [--save_loss_freq SAVE_LOSS_FREQ]
                                                      [--save_model_freq SAVE_MODEL_FREQ]
                                                      [--max_save MAX_SAVE]
                                                      [--show_loss SHOW_LOSS]
                                                      [--use_bias USE_BIAS]
                                                      [--size_factor SIZE_FACTOR]
                                                      [--kernel_network KERNEL_NETWORK [KERNEL_NETWORK ...]]
                                                      [--ecc_layers ECC_LAYERS]
                                                      [--pool_type POOL_TYPE]
                                                      [--activation ACTIVATION]
                                                      [--mlp_layers MLP_LAYERS]
                                                      [--log_name LOG_NAME]

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
  --save_model_dir SAVE_MODEL_DIR
                        directory to save the model
  --list_of_phases LIST_OF_PHASES [LIST_OF_PHASES ...]
                        map graph to phase, start from 0 to n-1
  --save_loss_freq SAVE_LOSS_FREQ
                        frequency to save the loss
  --save_model_freq SAVE_MODEL_FREQ
                        frequency to save the model
  --max_save MAX_SAVE   maximum number of checkpoints to keep
  --show_loss SHOW_LOSS
                        Show loss on the screen
  --use_bias USE_BIAS   use bias in the model
  --size_factor SIZE_FACTOR
                        increase/decrease dimensionality of hidden features
                        layer by layer of GNN/MLP
  --kernel_network KERNEL_NETWORK [KERNEL_NETWORK ...]
                        filter generating network size and hidden dimensions
                        graph
  --ecc_layers ECC_LAYERS
                        number of edge-conditioned convolutional layer
  --pool_type POOL_TYPE
                        type of pooling: sum, attn_sum, avg, attn_avg
  --activation ACTIVATION
                        activation function of layers: relu, tanh, ...
  --mlp_layers MLP_LAYERS
                        number of mlp layers after pooling
  --log_name LOG_NAME   write losses to this file
```

*Example*
Training
```
python main.py --list_of_graphs graph_Icehvapor_0.70_1.npz graph_Icehvapor_0.70_21.npz graph_Icehvapor_0.70_29.npz graph_Icehvapor_0.70_28.npz --dataset_size 40000 --batch_size 32 --learning_rate 0.00005 --epochs 50000 --list_of_phases 0 0 1 1
```
Prediction
```
python run.py # We will add further details for making it more user-friendly
```
