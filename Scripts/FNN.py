# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:56:35 2017
@author: viveksagar
"""
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#def create_attributes(calcium, spikes, num_history = 5, num_fut = 5, offset_step=5, num_der = 2, integ_kernel = 100):
## num_history = number of points in the past used as attribute, 
## num_fut = number of points in the future taken as attribute, offset_step = distance between history/future points
## num_der = maximum order of derivative, integ_kernel  = window size of running cumsum 
#    print("Creating additional attributes...")
#    cal_past = np.asarray([np.roll(np.squeeze(calcium),(ii+1)*offset_step) for ii in range(num_history)]).T # History
#    cal_fut = np.asarray([np.roll(np.squeeze(calcium),-(ii+1)*offset_step) for ii in range(num_fut)]).T # Future
#    cal_grad = np.asarray([np.append(np.diff(np.squeeze(calcium),ii+1),np.zeros(ii+1)) for ii in range(num_der)]).T # n-Gradients upto n-order      
#    cal_integ =  np.cumsum(calcium) # Cum-sum
#    cal_integ = (cal_integ[integ_kernel:]-cal_integ[:-integ_kernel])/integ_kernel # Moving Avg
#    cal_avg =  np.asarray([np.lib.pad(cal_integ, (len(calcium)-len(cal_integ),0), 'edge')]).T # Padded with additional values to maintain original size   
#    att_data_raw = np.concatenate((calcium,cal_past,cal_fut,cal_grad, cal_avg), axis = 1)     
#    mask = np.ones(len(att_data_raw),dtype=bool) # Clip the edges with inaccurate attributes
#    mask[:integ_kernel], mask[-num_fut*offset_step:] = False,False
#    att_data = att_data_raw[mask]
#    clipped_spikes = spikes[mask].astype(np.float64)    
#    return att_data, clipped_spikes
    
def create_attributes(calcium, spikes, num_history = 5, num_fut = 5, offset_step=5, num_der = 2, integ_kernel = 100):
# num_history = number of points in the past used as attribute, 
# num_fut = number of points in the future taken as attribute, offset_step = distance between history/future points
# num_der = maximum order of derivative, integ_kernel  = window size of running cumsum 
    print("Creating additional attributes...")
    cal_past = np.asarray([np.roll(np.squeeze(calcium),(ii+1)*offset_step) for ii in range(num_history)]).T # History
    cal_fut = np.asarray([np.roll(np.squeeze(calcium),-(ii+1)*offset_step) for ii in range(num_fut)]).T # Future
    cal_grad = np.asarray([np.append(np.diff(np.squeeze(calcium),ii+1),np.zeros(ii+1)) for ii in range(num_der)]).T # n-Gradients upto n-order      
    cal_integ =  np.cumsum(calcium) # Cum-sum
    cal_integ = (cal_integ[integ_kernel:]-cal_integ[:-integ_kernel])/integ_kernel # Moving Avg
    cal_avg =  np.asarray([np.lib.pad(cal_integ, (len(calcium)-len(cal_integ),0), 'edge')]).T # Padded with additional values to maintain original size   
    att_data_raw = np.concatenate((calcium,cal_past,cal_fut,cal_grad, cal_avg), axis = 1)     
    mask = np.ones(len(att_data_raw),dtype=bool) # Clip the edges with inaccurate attributes
    mask[:integ_kernel], mask[-num_fut*offset_step:] = False,False
    att_data = att_data_raw[mask]
    clipped_spikes = spikes[mask].astype(np.float64)    
    return att_data, clipped_spikes

    
def one_hot_label(label):
# One hot representation of labels for cross entropy measurement
    print("Using onehot representation")
    label_onehot = np.concatenate((label,1-label),axis=1)
    return label_onehot
    
def pre_train(att_data,clipped_spikes,batch_size):
#     Basic preprocessing prior to training: scaling, crossvalidation split, clipping the size to match batch size.
    att_data = preprocessing.scale(att_data)
    train_x, test_x, train_y, test_y = train_test_split(att_data, clipped_spikes, test_size=0.2)
    split_tr = (len(train_x)-np.mod(len(train_x),batch_size)).astype(np.int64)
    split_test = (len(test_x)-np.mod(len(test_x),batch_size)).astype(np.int64) 
    mask_tr = np.ones(len(train_x),dtype=bool) # Clip the edges to have the len of data a multiple of batch_size
    mask_test = np.ones(len(test_x),dtype=bool)
    mask_tr[split_tr:], mask_test[split_test:] = False,False
    train_x = train_x[mask_tr]
    train_y = train_y[mask_tr]
    train_y = one_hot_label(train_y)
    test_x = test_x[mask_test]
    test_y = test_y[mask_test]
    test_y = one_hot_label(test_y)
    return train_x, test_x, train_y, test_y
    
   
class FFN_architecture:
#     Create Tensorflow graph. Provide number of attributes (num_att) manually in mlpnet. Needs to be fixed.
    def __init__(self):
        self.batch_size = 100
        self.x = tf.placeholder(tf.float32) # Attributes
        self.y = tf.placeholder(tf.float32,shape=(self.batch_size,None)) # Labels
        self.o = self.mlpnet(self.x) # Predictions
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.o,labels=self.y))
        
    def mlp(self, input_, input_dim, output_dim, name):
#        It is possible to write this in a simpler form: Writing weights and bias of each layer. But this is more elegant. 
        with tf.variable_scope(name, reuse=None):
            w = tf.get_variable('w',[input_dim, output_dim], tf.float32, tf.random_normal_initializer(mean = 0.001,stddev=0.02))
            b = tf.get_variable('b',[1,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.00,stddev=0.02))
        return tf.nn.relu(tf.add(tf.matmul(input_,w),b))
        
    def mlpnet(self, x, num_att=14, _dropout= 0.5, nn1= 64, nn2=16, nn3=32):
        l1 = self.mlp(x,num_att,nn1,"l1")
        l1 = tf.nn.dropout(l1,_dropout)
        l2 = self.mlp(l1,nn1,nn2,"l2")
        l2 = tf.nn.dropout(l2,_dropout)
        l3 = self.mlp(l2,nn2,nn3,"l3")
        l3 = tf.nn.dropout(l3,_dropout)
        l4 = self.mlp(l3,nn3,2,"l4")
        return l4      
           
    
def train_neural_network(att_data, clipped_spikes, hm_epochs=25):
# Running the Tf session.    
    model = FFN_architecture()
    print('Model successfully acquired')    
    batch_size = model.batch_size 
    [train_x, test_x, train_y, test_y]=pre_train(att_data,clipped_spikes,batch_size)   
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(model.loss)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_x, model.y: batch_y})
                epoch_loss += c
                i+=batch_size
            print('Epoch', epoch+1, 'completed out of ', hm_epochs,'loss:',epoch_loss)
            spike_predict= model.o.eval({model.x:test_x})
#            sanity_check = model.y.eval({model.y:test_spikes})
            
    tf.reset_default_graph()
    return test_y, spike_predict

def main(calcium,spikes):
#    Provide inputs as 1D column vector of shape (N,1)
    [att_data,clipped_spikes] = create_attributes(calcium,spikes)
    [test_spikes, predicted_spikes]= train_neural_network(att_data,clipped_spikes)
    return test_spikes, predicted_spikes
    
