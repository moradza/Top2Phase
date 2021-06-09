from Top2Phase.utils import Top2PhaseDataset
from Top2Phase.model import PhaseModel
from Top2Phase.top2phase import Top2Phase
from spektral.transforms.normalize_adj import NormalizeAdj
#from Top2Phase.utils import *
import tensorflow as tf

tf.keras.backend.set_floatx('float32')

dataset = Top2PhaseDataset(1024*6, [['graph_Icehvapor_0.70_21.npz'], ['graph_Icehvapor_0.70_29.npz']],permute=False,  transforms=NormalizeAdj())

loader_tr = BatchLoader(dataset, batch_size=64, epochs=1000)                                                                                          
loader_te =  BatchLoader(dataset, batch_size=64, epochs=None)

model = PhaseModel()
top2phase = Top2Phase(model, loader_tr, loader_te, 0.00005, 2)
top2phase.train()
y_p, y_g = top2phas.predict(loader_te)
print(" 10 firts predictions, logits not classes")
print(y_p[:10])
print(y_g[:10])
