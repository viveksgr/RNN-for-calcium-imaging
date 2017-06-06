# -*- coding: utf-8 -*-
"""
@author: viveksagar
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import scipy.signal
import matplotlib.pyplot as plt

hm_epochs = 10
n_classes = 2
batch_size = 128
chunk_size = 20
n_chunks = 10
rnn_size = 64
drop = 0.8
n_folds=5 # Cross Validation
integ_kernel = 10 # Plotting

def create_attributes(calcium, spikes, num_history = 199, offset_step=1):
# num_history = number of points in the past used as attribute, 
    print("Creating additional attributes...")
    calcium = preprocessing.scale(calcium)
    cal_past = np.asarray([np.roll(np.squeeze(calcium),(ii+1)*offset_step) for ii in range(num_history)]).T # History
    att_data_raw = np.concatenate((calcium,cal_past), axis = 1)     
    mask = np.ones(len(att_data_raw),dtype=bool) # Clip the edges with inaccurate attributes
    mask[:num_history+1], mask[-num_history*offset_step:] = False,False
    att_data = att_data_raw[mask]
    clipped_spikes = spikes[mask].astype(np.float64)
    att_data = preprocessing.scale(att_data) 
    split = (len(att_data)-np.mod(len(att_data),batch_size*n_folds)).astype(np.int64)
    mask_tr = np.ones(len(att_data),dtype=bool) # Clip the edges to have the len of data a multiple of batch_size
    mask_tr[split:] = False
    att_data = att_data[mask_tr]
    clipped_spikes = clipped_spikes[mask_tr]     
    return att_data, clipped_spikes

def one_hot_label(label):
# One hot representation of labels for cross entropy measurement
    print("Using onehot representation")
    label_onehot = np.concatenate((label,1-label),axis=1)
    return label_onehot
    
# RNN    
def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}
    x = tf.unstack(x, n_chunks, 1)
    lstm_cell = rnn.BasicLSTMCell(rnn_size,forget_bias=1.0)
    
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.nn.softmax(tf.matmul(outputs[-1],layer['weights']) + layer['biases'])
    output = tf.nn.dropout(output,drop)
    return output
    
def train_neural_network(train_x, test_x, train_y, test_y):
    train_y = one_hot_label(train_y)
    test_y = one_hot_label(test_y)
    prediction = recurrent_neural_network(x) 
    cost = tf.div(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)),tf.reduce_sum(prediction))   
    correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_x = batch_x.reshape((batch_size,n_chunks,chunk_size))
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size
            print('Epoch', epoch+1, 'completed out of ', hm_epochs,'loss:',epoch_loss)
            test_x = test_x.reshape((-1,n_chunks,chunk_size))
            spike_predict= prediction.eval({x:test_x})
            acc = accuracy.eval({x: test_x, y: test_y})  
    tf.reset_default_graph()
    return spike_predict, acc    

def pcsn(clipped_spikes,prediction):
    count_real = np.where(clipped_spikes>0)[0]
    count_pred = np.where(prediction>0)[0]
    count_pred_margin = np.concatenate((count_pred,count_pred+2,count_pred-2))
    common = len(set(count_real).intersection(count_pred_margin))
    return common/len(count_real)         

def k_fold(calcium,spikes):
#    Provide inputs as 1D column vector of shape (N,1)
    [att_data,clipped_spikes] = create_attributes(calcium,spikes)
    kf = KFold(n_splits=n_folds,shuffle=False)
    X_tr,X_ts,Y_tr,Y_ts = [],[],[],[]
    for train_index, test_index in kf.split(att_data):
        train_x, test_x = att_data[train_index], att_data[test_index]
        train_y, test_y = clipped_spikes[train_index], clipped_spikes[test_index]
        X_tr.append(train_x)
        X_ts.append(test_x)
        Y_tr.append(train_y)
        Y_ts.append(test_y)
    return X_tr,X_ts,Y_tr,Y_ts
    
def smoothen(raw_dat,n_smooths=5):
    this_medfilt=3
    smooth_mat=np.zeros((n_smooths,raw_dat.shape[0]))
    for n in range(n_smooths):
        smooth_mat[n,:]=scipy.signal.medfilt(np.squeeze(raw_dat),this_medfilt)
        this_medfilt+=2
    smooth_mat = np.mean(smooth_mat,axis=0)
    return smooth_mat
    
def cum_sum(vec):
    cal_integ =  np.cumsum(vec) # Cum-sum
    cal_integ = cal_integ[integ_kernel:]-cal_integ[:-integ_kernel]
    return cal_integ
    
smooth_calcium = smoothen(calcium).reshape(-1,1)
X_tr,X_ts,Y_tr,Y_ts = k_fold(smooth_calcium,spikes)
prediction = []
accuracy = []
for fold in range(n_folds):   
    x = tf.placeholder('float', [None, n_chunks,chunk_size])
    y = tf.placeholder('float')
    [predicted_spikes, acc]= train_neural_network(X_tr[fold], X_ts[fold], Y_tr[fold], Y_ts[fold])
    prediction.append(predicted_spikes)
    accuracy.append(acc)   
prediction = np.vstack(prediction)[:,1]
clipped_spikes = np.vstack(Y_ts)[:,0]
accuracy = np.mean(accuracy)
calcium_plt = smoothen(np.vstack(X_ts)[:,0])
prediction[prediction>1]=1
prediction[prediction<1]=0
prediction = 1-prediction
precision = pcsn(clipped_spikes, prediction)

cfm = confusion_matrix(clipped_spikes,prediction)/len(prediction)

num_pt = 1000
plt.figure(figsize=(20,5))
a=plt.vlines(np.arange(num_pt)/1000,0,(prediction[0:num_pt+integ_kernel]),'r',label='Prediction')
b=plt.vlines(np.arange(num_pt)/1000,0,(clipped_spikes[0:num_pt+integ_kernel]),'b',label='Actual Spikes')
c = plt.plot(np.arange(num_pt)/1000,calcium_plt[integ_kernel:num_pt+integ_kernel]-np.min(calcium_plt),'k',label='Calcium Trace (unscaled)')
plt.xlabel('Time(s)')
plt.ylabel('# Spikes in 10 ms window')
plt.ylim([0,1])
plt.xlim([0,1])
plt.legend()
#plt.show()
plt.savefig('plot.png')

    
