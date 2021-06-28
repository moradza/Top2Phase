import numpy as np
import os
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow as tf
from spektral.layers.pooling import global_pool

from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.layers import ECCConv


class PhaseModel(Model):
    def __init__(self, 
                Node_dim=2, 
                Edge_dim=4, 
                Output_dim=1, 
                kernel_network=[30,60,30], 
                ecc_layers=4, 
                ecc_hidden_factor=2,
                fix_size=[32,64,32,32,30,15,9],
                mlp_layers=3,
                pool_type='sum',
                activation='relu',
                use_bias=True):
        # Store all networks architecture in config
        self.config = {'Node_dim': Node_dim,
                       'Edge_dim': Edge_dim,
                       'Output_dim': Output_dim,
                       'kernel_network': kernel_network,
                       'ecc_layers': ecc_layers,
                       'ecc_hidden_factor': ecc_hidden_factor,
                       'mlp_layers': mlp_layers,
                       'pool_type': pool_type,
                       'activation':activation,
                       'use_bias': use_bias,
                       'fix-size': fix_size}
        super(PhaseModel, self).__init__()
        # New network with fixed size
        self.ECCNet = []
        if not fix_size is None:
            assert len(fix_size) == mlp_layers + ecc_layers
            assert fix_size[-1] == Output_dim
            
            self.GNNSize = fix_size[:ecc_layers]
            self.MLPSize = fix_size[ecc_layers:]
        else:
            self.GNNSize =[ int(Node_dim*ecc_hidden_factor**(cnt_layer+1)) for cnt_layer in range(ecc_layers)]
            self.MLPSize =[ max(Output_dim, int(Node_dim*ecc_hidden_factor**(ecc_layers-cnt_layer-1))) if cnt_layer < mlp_layers-1 else Output_dim for cnt_layer in range(mlp_layers) ]
            
        ################################################################################
        # CREATE NETWORK
        ################################################################################
        #self.ECCNet = [] 
        # [ECCConv(int(self.Edge_dim*ecc_hidden_factor**(i+1), kernel_network=self.kernel_network, activation="relu")for i in range(ecc_layer)]
        print('information about the model: ')
        for cnt_layer in range(ecc_layers):
            self.ECCNet.append(ECCConv(self.GNNSize[cnt_layer], use_bias=use_bias,\
                                       kernel_network=kernel_network, activation=activation))
            print("hidden dimensions of layer : ", cnt_layer+1, " of GNN equals to : ", self.GNNSize[cnt_layer])
        
        # Add pooling layer
        self.pool = global_pool.get(pool_type)()
        
        # Add MLP layers 
        self.MLPNet = []
        self.BNs = []
        for cnt_layer in range(mlp_layers-1):
            self.MLPNet.append(Dense(self.MLPSize[cnt_layer],use_bias=use_bias))# activation=activation))
            #self.BNs.append(tf.keras.layers.BatchNormalization())
            #if self.batch_norm and cnt_layer < mlp_layers - 2:
            #    self.MLPNet.append(tf.keras.layers.BatchNormalization())
            print("hidden dimensions of layer : ", cnt_layer+1, ' of MLP equals to : ', self.MLPSize[cnt_layer])
         
        # output layer with no activation
        self.MLPNet.append(Dense(self.MLPSize[-1]))
        
        # save model configuration
        np.savez('Model.config.npz', **self.config) 
        
        self.non_functional = False
        if self.non_functional:
            pass

    def get_config(self):
        return self.config
    
    def call(self, inputs,training=True):
        x, a, e = inputs
        
        for ecc_layer in range(self.config['ecc_layers']):
            x = self.ECCNet[ecc_layer]([x,a,e])
        
        x = self.pool(x)
        
        for mlp_layer in range(self.config['mlp_layers']-1):
            x = self.MLPNet[mlp_layer](x)
            #x = self.BNs[mlp_layer](x,training=training)
            x = tf.nn.relu(x)
            #if self.batch_norm and mlp_layer < self.config['mlp_layers']-2:
            #x = self.MLPNet
        return self.MLPNet[-1](x)
