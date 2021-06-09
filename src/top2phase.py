import numpy as np
import sys
import os
from spektral.data import Dataset, BatchLoader
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

class Top2Phase:
    def __init__(self,
                 model,
                 data_loader_tr,
                 data_loader_val,
                 learning_rate,
                 num_classes,
                 save_model_dir='./tf_ckpts',
                 num_model_ckpts=5,
                 save_model_freq=10,
                 save_loss_freq=10,
                 log_name='training.log',
                 show_loss=True):
        self.model = model
        self.loader_tr = data_loader_tr
        self.loader_val = data_loader_val
        self.learning_rate = learning_rate
        # *** Implement early stopping *** #
        
        # Input validation
        assert type(num_classes) == int
        assert  num_classes >= 1
        assert isinstance(self.loader_tr, BatchLoader)
        assert isinstance(self.loader_val, BatchLoader)
        assert isinstance(save_loss_freq, int)
        assert isinstance(save_model_freq, int)
        assert isinstance(learning_rate, float)
        assert learning_rate > 0.0
        
        # Identify loss function and accuracy metric
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        if num_classes > 2:
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            #self.accuracy = tf.keras.metrics.CategoricalAccuracy(name='Categorical_accuracy', dtype=None) 
            #self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy') 
        else:
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            #self.accuracy = tf.keras.metrics.BinaryAccuracy(name='Binary_accuracy', threshold=0.0)
            self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy', threshold=0.0)
            self.val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy', threshold=0.0)
        # Optimizer and log file names
        self.opt = Adam(lr=self.learning_rate)
        self.log_name = log_name
        
        # Model save and restore variables
        self.save_model_dir = save_model_dir
        self.num_model_ckpts = num_model_ckpts
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.opt, net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.save_model_dir, max_to_keep=self.num_model_ckpts)
        self.save_model_freq = save_model_freq
        self.save_loss_freq = save_loss_freq
        self.show_loss = show_loss
    
    def predict(self, data_loader):
        assert isinstance(data_loader, BatchLoader)
        # restore model
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(" ############################################## ")
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing model from scratch.")
        y_pred = []
        y_truth = []
        cur_batch = 0
        for batch in data_loader:
            y_pred.extend(self.model(batch[0]).numpy())
            y_truth.extend(batch[1])
            if cur_batch == data_loader.steps_per_epoch:
                break
            cur_batch += 1
        y_pred = np.array(y_pred)
        y_truth = np.array(y_truth)
        return y_pred, y_truth
        
    def write_loss(self, loss_tr, loss_val, acc_tr, acc_val):
        if not os.path.isfile(self.log_name):
            with open(self.log_name,'a') as f:
                f.write('#epochs loss_tr acc_tr loss_val acc_val'+"\n")
                f.close()
        with open(self.log_name,'a') as f:
            f.write(str(int(self.ckpt.step))+ ' '+str(loss_tr)+' '+str(acc_tr)+ ' '\
                    + str(loss_val)+ ' ' + str(acc_val)+"\n")
            f.close()
        
    def evaluate(self, data_loader):
        assert isinstance(data_loader, BatchLoader)
        model_loss = 0
        model_acc = 0
        cur_batch = 0
        for batch in data_loader:
            inputs, target = batch
            predictions = self.model(inputs, training=False)
            v_loss = self.loss_fn(target, predictions)
            self.val_loss(v_loss)
            self.val_accuracy(target, predictions)
            cur_batch += 1
            if cur_batch == data_loader.steps_per_epoch:
                break
        #model_loss /= data_loader.steps_per_epoch
        #model_acc /= data_loader.steps_per_epoch 
        #return model_loss, model_acc

    def train(self):
        @tf.function(input_signature=self.loader_tr.tf_signature(), experimental_relax_shapes=True)
        def train_step(inputs, target):
            loss, acc = 0.0, 0.0
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss = self.loss_fn(target, predictions)
                #loss += sum(self.model.losses)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
            #acc = self.accuracy(target, predictions)
            self.train_loss(loss)
            self.train_accuracy(target, predictions)

        cur_batch = 0
        tr_loss, tr_acc = 0.0, 0.0
        cur_epoch = 0
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(" ############################################## ")
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing model from scratch.")
        self.train_loss.reset_states()
        self.val_loss.reset_states()
        self.train_accuracy.reset_states()
        self.val_accuracy.reset_states()

        for batch in self.loader_tr:
            #self.train_loss.reset_states()
            #train_accuracy.reset_states()
            #self.val_loss.reset_states()
            #test_accuracy.reset_states()
            train_step(*batch)
            #tr_loss += outs[0].numpy()
            #tr_acc += outs[1].numpy()
            cur_batch += 1
            if cur_batch == self.loader_tr.steps_per_epoch:
                cur_batch = 0
                self.evaluate(self.loader_val)
                print(
                     f'Epoch {int(self.ckpt.step)}, '
                     f'Loss: {self.train_loss.result()}, '
                     f'Accuracy: {self.train_accuracy.result() * 100}, '
                     f'Test Loss: {self.val_loss.result()}, '
                     f'Test Accuracy: {self.val_accuracy.result() * 100}'
                     )
                if int(self.ckpt.step) % self.save_loss_freq == 0 :
                    self.write_loss(self.train_loss.result().numpy(), self.train_accuracy.result().numpy(),\
                                self.val_loss.result().numpy(), self.val_accuracy.result().numpy())
                self.train_loss.reset_states()
                self.val_loss.reset_states()
                self.train_accuracy.reset_states()
                self.val_accuracy.reset_states()
                if int(self.ckpt.step) % self.save_model_freq  == 0:
                    save_path = self.manager.save()
                #    val_loss, val_acc =  self.evaluate(self.loader_val)
                #    tr_loss /= self.loader_tr.steps_per_epoch
                #    tr_acc /= self.loader_tr.steps_per_epoch
                #if int(self.ckpt.step) % self.save_loss_freq == 0 :
                #    self.write_loss(self.train_loss.result(), self.train_accuracy.result(),\
                #                self.val_loss.result(), self.val_accuracy.result())
                #if self.show_loss:
                #        print('epoch: ' + str(int(self.ckpt.step)) + \
                #                         ' , train loss: ' + str(tr_loss) + ' , train acc: ' + str(tr_acc) +\
                #                         ' , val loss: ' + str(val_loss.numpy()) + ' , val acc: ' + str(val_acc.numpy()))
                #tr_loss , tr_acc = 0.0 , 0.0
                self.ckpt.step.assign_add(1)
