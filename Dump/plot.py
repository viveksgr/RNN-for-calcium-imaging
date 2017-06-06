# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:03:06 2017

@author: viveksagar
"""

from copy import deepcopy as cp

pred = predicted_spikes[:,1]
pred[pred>0]=1
pred = 1-pred

pred2 = cp(pred)
for ii in range(len(pred)-1):
    if ii>0:    
        if pred[ii]==1:
            pred2[ii+1]=1
            pred2[ii-1]=1
            
test = test_spikes[:,0]

count = 2*test-pred2
pcn = len(np.where(count>0)[0])/np.sum(test) 