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


class FFN:
    def __init__(self, nn1=10, nn2 = 5, nn3 = 5):
        self.x = tf.placeholder(tf.float32)
        self.o = self.mlpnet(self.x)
        self.y = tf.placeholder(tf.float32)      
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.o,self.y))
        
        self.hidden_1_layer = {'f_fum':nn1,'weight':tf.Variable(tf.random_normal([len(self.x[0]), nn1])),'bias':tf.Variable(tf.random_normal([nn1]))}
        self.hidden_2_layer = {'f_fum':nn2,'weight':tf.Variable(tf.random_normal([nn1, nn2])),'bias':tf.Variable(tf.random_normal([nn2]))}
        self.hidden_3_layer = {'f_fum':nn3,'weight':tf.Variable(tf.random_normal([nn2, nn3])),'bias':tf.Variable(tf.random_normal([nn3]))}
        self.output_layer = {'f_fum':None,'weight':tf.Variable(tf.random_normal([nn3, 1])),'bias':tf.Variable(tf.random_normal([1]))}

    def mlpnet(self,data,drop=0.7):
        l1 = tf.add(tf.matmul(data,self.hidden_1_layer['weight']), self.hidden_1_layer['bias'])
        l1 = tf.nn.relu(l1)
        l1 = tf.nn.dropout(l1,drop)
        l2 = tf.add(tf.matmul(l1,self.hidden_2_layer['weight']), self.hidden_2_layer['bias'])
        l2 = tf.nn.relu(l2)
        l2 = tf.nn.dropout(l2,drop)
        l3 = tf.add(tf.matmul(l2,self.hidden_3_layer['weight']), self.hidden_3_layer['bias'])
        l3 = tf.nn.relu(l3)
        l3 = tf.nn.dropout(l3,drop)
        output = tf.nn.softsign(tf.matmul(l3,self.output_layer['weight']) + self.output_layer['bias'])
        return output
    
def train_neural_network(att_data, clipped_spikes, hm_epochs=2, batch_size= 100):
    att_data = preprocessing.scale(att_data)
    train_x, test_x, train_y, test_y = train_test_split(att_data, clipped_spikes, test_size=0.2)    
    model = FFN()
    print ('Model acquired')
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
    return test_y, spike_test
