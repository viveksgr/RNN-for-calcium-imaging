# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:56:35 2017
@author: viveksagar
"""
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def create_attributes(calcium, spikes, num_history = 5, num_fut = 5, offset_step=5, num_der = 2, integ_kernel = 50):
# num_history = number of points in the past used as attribute, 
# num_fut = number of points in the future taken as attribute, offset_step = distance between history/future points
# num_der = maximum order of derivative, integ_kernel  = window size of running cumsum 
    print("Creating additional attributes...")
    cal_past = np.asarray([np.roll(np.squeeze(calcium),-(ii+1)*offset_step) for ii in range(num_history)]).T # History
    cal_fut = np.asarray([np.roll(np.squeeze(calcium),(ii+1)*offset_step) for ii in range(num_fut)]).T # Future
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


class FFN_architecture:
    def __init__(self):
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.o = self.mlpnet(self.x)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.o,self.y))

        
    def mlp(self, input_, input_dim, output_dim, name):
        with tf.variable_scope(name, reuse=None):
            w = tf.get_variable('w',[input_dim, output_dim], tf.float32, tf.random_normal_initializer(mean = 0.001,stddev=0.02))
            b = tf.get_variable('b',[1,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.00,stddev=0.02))
        return tf.nn.softsign(tf.add(tf.matmul(input_,w),b))
        
    def mlpnet(self, x,_dropout= 0.7, nn1= 10, nn2=10, nn3=10, num_att =14):
        l1 = self.mlp(x,num_att,nn1,"l1")
        l1 = tf.nn.dropout(l1,_dropout)
        l2 = self.mlp(l1,nn1,nn2,"l2")
        l2 = tf.nn.dropout(l2,_dropout)
        l3 = self.mlp(l2,nn2,nn3,"l3")
        l3 = tf.nn.dropout(l3,_dropout)
        l4 = self.mlp(l3,nn3,1,"l4")
        return l4       
           
    
def train_neural_network(att_data, clipped_spikes, hm_epochs=4, batch_size= 100):
    att_data = preprocessing.scale(att_data)
    train_x, test_x, train_y, test_y = train_test_split(att_data, clipped_spikes, test_size=0.2)    
    model = FFN_architecture()
    print('Model successfully acquired')
    loss = model.loss
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
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
                _, c, spike = sess.run([optimizer, model.loss, model.o], feed_dict={model.x: batch_x, model.y: batch_y})
                epoch_loss += c
                i+=batch_size
            print('Epoch', epoch+1, 'completed out of ', hm_epochs,'loss:',epoch_loss)
            spike_test= model.o.eval({model.x:test_x})
    tf.reset_default_graph()
    return test_y, spike_test
