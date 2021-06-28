from utils import *
from model import *
from top2phase import *
from gnnexplainer import *

tf.keras.backend.set_floatx('float64')

dataset = Top2PhaseDataset(2000, [['graph_Icehvapor_0.70_5.npz'], ['graph_Icehvapor_0.70_21.npz']],permute=True,  transforms=NormalizeAdj())

loader_tr = BatchLoader(dataset, batch_size=2, epochs=1)                                                                                          

#loader_te =  BatchLoader(dataset, batch_size=2, epochs=None)
for b in loader_tr:
    inputs, t = b

model = PhaseModel()
top2phase = Top2Phase(model, loader_tr, loader_tr, 0.00005, 2)
top2phase.ckpt.restore(top2phase.manager.latest_checkpoint)
 
gnnexplainer= GNNExplainer(top2phase.model,preprocess=NormalizeAdj(),learning_rate=0.01, a_size_coef=0.02, a_entropy_coef=0.0002, x_size_coef=0.0, x_entropy_coef=0.0)
x, y = gnnexplainer.explain_node(inputs[0][0].reshape((-1,inputs[0][0].shape[0],2)), inputs[1][0].reshape((1,inputs[1].shape[1],inputs[1].shape[1])),inputs[2][0].reshape((1,inputs[2].shape[1],inputs[2].shape[1],1)),epochs=20000) 

x = tf.nn.sigmoid(x)



#from model import *
#from top2phase import *
#model = PhaseModel()
#top2phase = Top2Phase(model, loader_tr, loader_te, 0.00005, 2)
#top2phase.train()

#model = tf.keras.models.load_model('./GCN_model.h5',custom_objects={'ECCConv': ECCConv, 'GlobalSumPool':global_pool.get('sum')})
'''
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
opt = Adam(lr=0.001) 
accuracy = tf.keras.metrics.BinaryAccuracy(name='Binary_accuracy', threshold=0.0)

@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    acc = accuracy(target, predictions)
    return loss, acc
def evaluate():
    print("Testing model")
    model_loss = 0
    model_acc = 0
    cur_batch = 0
    for batch in loader_te:
        inputs, target = batch
        predictions = model(inputs, training=False)
        model_loss += loss_fn(target, predictions)
        model_acc += accuracy(target, predictions)
        if cur_batch == loader_te.steps_per_epoch:
            break
        cur_batch += 1
    model_loss /= loader_te.steps_per_epoch
    model_acc /= loader_te.steps_per_epoch
    print("Done. Test loss: {}, Acc: {}".format(model_loss, model_acc))
'''
#model.build(input_signature=loader_tr.tf_signature())
#csv_logger = tf.keras.callbacks.CSVLogger('training.log',append=True)
#model.compile(opt, loss_fn)
#model.build(input_shape=loader_tr.tf_signature())
#model.summary()
#model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, epochs=100, callbacks=[csv_logger])
#model.build(loader_tr.tf_signature())

#for batch in loader_tr:
#    outs = train_step(*batch)
#    model.load_weights('./GCN_model4.h5')
#    break
#new_model.build((None, *x_train.shape[1:]))
'''
print("Fitting model")
cur_batch = 0
model_loss = 0
for batch in loader_tr:
    outs = train_step(*batch)
    #print("loss: ", outs[0].numpy(), outs[1].numpy())
    cur_batch += 1
    if cur_batch == loader_te.steps_per_epoch:
        evaluate()
        cur_batch = 0
    model.save_weights('./GCN_model4.h5')
'''
