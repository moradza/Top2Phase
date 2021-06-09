import os
import argparse
from utils import *
from model import *
from top2phase import *

parser = argparse.ArgumentParser('ECCConv net training  for phase classification')
parser.add_argument('--list_of_graphs', nargs='+', default=['graph_Icehvapor_0.70_21.npz', 'graph_Icehvapor_0.70_29.npz'], help='list of graphs to use for training save as npz,\
                                                                     \n it should be generated using the convert_to_graph.py') # add datatset path as another parser, perform sanity check for consistent pass of graph names, add a dictioanry (k,v) :(graph_name, class)
parser.add_argument('--dataset_size', type=int,  default=1024*6,  help=' dataset size (int)') #check for dataset <= len(graphs)
parser.add_argument('--learning_rate', type=float, default=0.00005, help=' learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for each epochs')
parser.add_argument('--epochs', type=int, default=1000, help='total number of training epochs')
parser.add_argument('--data_path', default=os.getcwd(), help='path to the data file')
parser.add_argument('--random_seed', default=12345, type=int, help='random seed for consistent training')
parser.add_argument('--split_fraction', default=0.2, type=float, help='fraction of data for validation')
parser.add_argument('--save_model_dir', default='./tf_ckpts', help='directory to save the model', required=False)
parser.add_argument('--list_of_phases', default=[0,1], nargs='+',type=int, help='map graph to phase, start from 0  to n-1',required=False)
parser.add_argument('--save_loss_freq', default=10, type=int, help='frequency to save the loss')
parser.add_argument('--save_model_freq', default=10, type=int, help='frequency to save the model')
parser.add_argument('--max_save', default=5, type=int, help='maximum number of checkpoints to keep')
parser.add_argument('--show_loss', default=True, type=bool, help='Show loss on the screen')
parser.add_argument('--use_bias', default=True, type=bool, help='use bias in the model')
parser.add_argument('--size_factor', default=2, type=int, help='increase/decrease dimensionality of hidden features layer by layer of GNN/MLP')
parser.add_argument('--kernel_network', default=[30,60,30], nargs='+',type=int, help='filter generating network size and hidden dimensions graph')
parser.add_argument('--ecc_layers', default=4, type=int, help='number of edge-conditioned convolutional layer')
parser.add_argument('--pool_type', default='sum', type=str, help='type of pooling: sum, attn_sum, avg, attn_avg')
parser.add_argument('--activation', default='relu', type=str, help='activation function of layers: relu, tanh, ...')
parser.add_argument('--mlp_layers', default=4, type=int, help='number of mlp layers after pooling')
parser.add_argument('--log_name', default='training.log', type=str, help='write losses to this file')
args = parser.parse_args()

np.random.seed(args.random_seed)
print("list of graph files: ", args.list_of_graphs)
if args.split_fraction <= 0.0 or args.split_fraction >= 1.0:
    raise ValueError('Split fraction should lie between 0 and 1')
if not os.path.isdir(args.data_path):
    raise NameError(data_path +' is not valid')
if len(args.list_of_graphs) < 2:
    raise NameError('At least two graphs should be provided')
for kernel in args.kernel_network:
    if kernel < 1:
        raise ValueError('Kernel network should have a dimension larger than 1')
 
################################################################################
# GRAPHS SIZE AND LIST
################################################################################
list_of_phases = args.list_of_phases
if len(list_of_phases) != len(args.list_of_graphs):
    raise NameError("list of phase and graphs should be of the same length!")
if np.unique(list_of_phases).shape[0] < 2:
    raise NameError("list of phase at list should have to unique elements!")

list_of_graphs = []
list_of_graphs = [[args.list_of_graphs[i] for i in range(len(args.list_of_graphs)) if list_of_phases[i] == j ]\
                   for j in range(np.unique(list_of_phases).shape[0])]

dataset_size = args.dataset_size

################################################################################
# PARAMETERS
################################################################################
learning_rate = args.learning_rate  # Learning rate
epochs = args.epochs  # Number of training epochs
batch_size = args.batch_size  # Batch size

################################################################################
# LOAD And split DATA 
# Frequency to save fit data for now 1 is good enough
################################################################################
dataset = Top2PhaseDataset(dataset_size, list_of_graphs, rseed=args.random_seed,\
                    data_path=args.data_path, permute=False, transforms=NormalizeAdj())
split = int(args.split_fraction * dataset_size)

loader_tr = BatchLoader(dataset[:-split], batch_size=batch_size, epochs=epochs)
loader_val = BatchLoader(dataset[-split:], batch_size=batch_size, epochs=None)#int(epochs/(1-args.split_fraction)))

# Parameters
F = 2  # Dimension of node features
S = 1 #dataset.n_edge_features  # Dimension of edge features
n_out = len(list_of_graphs) if len(list_of_graphs) > 2 else 1# dataset.n_labels  # Dimension of the target

################################################################################
# BUILD MODEL, Network, Optimizer, Loss Function
################################################################################
model = PhaseModel(Node_dim=F,
                Edge_dim=S,
                Output_dim=n_out,
                kernel_network=args.kernel_network,
                ecc_layers=args.ecc_layers,
                ecc_hidden_factor=args.size_factor,
                mlp_layers=args.mlp_layers,
                pool_type=args.pool_type,
                activation=args.activation,
                use_bias=args.use_bias)

top2phase = Top2Phase(model,
                 loader_tr,
                 loader_val,
                 learning_rate,
                 n_out,
                 save_model_dir=args.save_model_dir,
                 num_model_ckpts=args.max_save,
                 save_model_freq=args.save_model_freq,
                 save_loss_freq=args.save_loss_freq,
                 log_name=args.log_name,
                 show_loss=args.show_loss)
top2phase.train()

