# Top2Phase
Topological Classification of Water Phases Using Edge-Conditioned Convolutional Graph Neural Network  


Python codes are located in the src/. directory:


1. convert_traj_to_graph.py : graphical user interface supports user interface and communication between various codes
2. ulities.py: 
3. utility.py : kernele ridge regression is implemented for MAVELP dataset and used for fitting, prediction and score, additionally it supports optimization
4. model.py : neural network model for classificaiton is implemented. 
5. main.py : training of the network model for classificaiton is done in this part. 

## Installation
  We recommend installaiton of following packages for experimentation with this code, preferabely all in a conda enviroment
  ```
  conda create -n top2phase python=3.6 tensorflow pymatgen tqdm -c anaconda -c conda-forge
  source activate top2phase 
  pip install pyboo
  pip install spektral==1.0.4
  
  ```
  ### Requirment
    tensorflow >= 2.1 
    spektral >= 1.0.4 
    pyboo  
    pymatgen 
    tqdm 
## Usage

### Trajectory to Graph



### Graph Neural network training
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
