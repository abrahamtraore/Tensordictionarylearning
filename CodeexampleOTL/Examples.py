#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:17:54 2019

@author: Traoreabraham
"""
import numpy as np
from multiprocessing import Pool

import sys
sys.path.append("..")

#sys.path.append("/home/scr/etu/sil821/traorabr/Tensorly/")
import tensorly as tl
backendchoice='numpy'
tl.set_backend(backendchoice)


import pdb

np.random.seed(6)


from MethodsTSPen import  pool_init

from OnlineTensor import CyclicBlocCoordinateTucker_setWithPredefinedEpochs

#backend choice for Tensorly (see Tensorly for more precisions)


#Examples for the OTL approach
K=20#the training set is compsed of 20 samples
Tensor_trainset=[]
Pre_existingGtrainset=[]
for k in range(K):
    Tensor_trainset.append(tl.tensor(np.maximum(np.random.normal(loc=0,scale=1,size=(20,30,40)),0)))
    Pre_existingGtrainset.append(tl.tensor(np.maximum(np.random.normal(loc=0,scale=1,size=(10,15,20)),0)))

Coretensorsize=np.array([10,15,20])
Pre_existingfactors=[tl.tensor(np.random.rand(20,10)),tl.tensor(np.random.rand(30,15)),tl.tensor(np.random.rand(40,20))]
#Dictionnary matrices
Pre_existingP=[tl.tensor(np.random.rand(20,10)),tl.tensor(np.random.rand(30,15)),tl.tensor(np.random.rand(40,20))]
#These matrices Pt have exactly the same sizes as the dictionary 
Pre_existingQ=[tl.tensor(np.random.rand(10,10)),tl.tensor(np.random.rand(15,15)),tl.tensor(np.random.rand(20,20))]
#These matrices are square and the dimensions are given by the core tensor sizes
Nonnegative=True
#True for the positivity constraints
#"Ortho" for the orthogonality constraints (for orthogonality, be aware of the dimensions: each dimension for the core must be smaller than the corresponding dimension for the observations)
Reprojectornot=True
#True if you want to recompute the activation coefficients. If you want to recover the activation tensor, set this variable to True. 
#In this case, the functions returns two lists: the first for the activation tensors, the second for the dictionary matrices
Setting="Single" #This means you are interested in using OTLsingle (instead of OTLminibatch)
Minibatchsize=[]#Since you decide to use OTLsingle, this parameter will not be used at all
step=np.power(10,-6,dtype=float)
alpha=np.power(10,1,dtype=float)
theta=np.power(10,-1,dtype=float)#This parameter must be strictly inferior to 1
max_iter=10
epsilon=np.power(10,-6,dtype=float)
period=3# this mean that the golden method is used each three iterations to determine the descent step
nbepochs=1#The number of epochs (number of times each sample is used)
pool=Pool(5,initializer=pool_init(),maxtasksperchild=100000000)
      
listoffactors=CyclicBlocCoordinateTucker_setWithPredefinedEpochs(Tensor_trainset,Coretensorsize,Pre_existingfactors,Pre_existingGtrainset,Pre_existingP,Pre_existingQ,Nonnegative,Reprojectornot,Setting,Minibatchsize,step,alpha,theta,max_iter,epsilon,period,nbepochs,pool)
  

#Examples for the OTLminibatch approach
#K=10#the training set is compsed of 10 samples
K=10
Tensor_trainset=[]
Pre_existingGtrainset=[]
for k in range(K):
    Tensor_trainset.append(tl.tensor(np.maximum(np.random.normal(loc=0,scale=1,size=(20,30,40)),0)))
    Pre_existingGtrainset.append(tl.tensor(np.maximum(np.random.normal(loc=0,scale=1,size=(10,15,20)),0)))

Coretensorsize=np.array([10,15,20])
Pre_existingfactors=[tl.tensor(np.random.rand(20,10)),tl.tensor(np.random.rand(30,15)),tl.tensor(np.random.rand(40,20))]
#Dictionnary matrices
Pre_existingP=[tl.tensor(np.random.rand(20,10)),tl.tensor(np.random.rand(30,15)),tl.tensor(np.random.rand(40,20))]
#These matrices Pt have exactly the same sizes as the dictionary 
Pre_existingQ=[tl.tensor(np.random.rand(10,10)),tl.tensor(np.random.rand(15,15)),tl.tensor(np.random.rand(20,20))]
#These matrices are square and the dimensions are given by the core tensor sizes
Nonnegative=True
#True for the positivity constraints
#"Ortho" for the orthogonality constraints (for orthogonality, be aware of the dimensions: each dimension for the core must be smaller than the corresponding dimension for the observations)
Reprojectornot=False
#True if you want to recompute the activation coefficients. If you want to recover the activation tensor, set this variable to True. 
###In this case, the functions returns two lists: the first for the activation tensors, the second for the dictionary matrices
Setting="MiniBatch"#This means you are interested in using OTLsingle (instead of OTLminibatch)
Minibatchsize=[5,3,2]#The sum of these number must be equal to the cardinalité of the training set, which is here 10
step=np.power(10,-6,dtype=float)
alpha=np.power(10,1,dtype=float)
theta=np.power(10,-1,dtype=float)#This parameter must be strictly inferior to 1
max_iter=10
epsilon=np.power(10,-6,dtype=float)
period=3# this mean that the golden method is used each three iterations to determine the descent step
nbepochs=1#The number of epochs (number of times each sample is used)
pool=Pool(5,initializer=pool_init(),maxtasksperchild=100000000)
##      
#listoffactors=CyclicBlocCoordinateTucker_setWithPredefinedEpochs(Tensor_trainset,Coretensorsize,Pre_existingfactors,Pre_existingGtrainset,Pre_existingP,Pre_existingQ,Nonnegative,Reprojectornot,Setting,Minibatchsize,step,alpha,theta,max_iter,epsilon,period,nbepochs,pool)   
pdb.set_trace()   