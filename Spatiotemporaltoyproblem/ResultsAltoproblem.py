#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:28:25 2018

@author: Traoreabraham
"""

import numpy as np
import pdb
from multiprocessing import Pool
import sys
import pdb
import scipy.io
#sys.path.append("/home/scr/etu/sil821/traorabr/OnlineTensorDictionaryLearning/")
sys.path.append("..")
import CodingAltoProblem
np.random.seed(1)

def Lowrankdefinitionproblem(P,T,Q,M,rank,Y,m):
    np.random.seed(m)
    W=np.zeros(P*Q*M)
    for r in range(rank):
       a=np.random.rand(P)
       b=np.random.rand(Q)
       c=np.random.rand(M)
       W=W+(r+1)*np.kron(np.kron(a,b),c)
    W=np.reshape(W,(P,Q,M))
    X=np.zeros((P,T,M))
    for m in range(M):
        X[:,:,m]=np.dot(W[:,:,m],Y[:,:,m])+np.random.normal(loc=0,scale=1/2,size=(P,T))
    return X

P=56
T=1200
Q=56
M=15
rank=2
mu=0.1
RMSE=[]
for m in range(10):
    rank=2
    seed=10
    Y=np.random.normal(loc=0,scale=1/2,size=(Q,T,M))
    X=Lowrankdefinitionproblem(P,T,Q,M,rank,Y,seed)
    A=np.random.normal(loc=0,scale=1/2,size=(P,P))
    Similaritymatrix=np.dot(A,A.T)+mu*np.eye(P)
    H=np.linalg.cholesky(Similaritymatrix)
    adress='/Users/Traoreabraham/Desktop/OnlineTensorDictionaryLearning/ALTO_Code/SpatiotemporaltoyproblemToydata'
    scipy.io.savemat(adress+str(m+1)+'.mat', dict(Data=X, CholsimHy=H))    
#    nLag=3
#    Predictor,Response=CodingAltoProblem.series_to_samples(X,nLag)
#  
#    T=np.array(Predictor.shape,dtype=int)[1]
#    train_ratio=0.9
#    Predictortrain=Predictor[:,0:int(train_ratio*T),:]
#    Responsetrain=Response[:,0:int(train_ratio*T),:]
#    Predictortest=Predictor[:,int(train_ratio*T):T,:]
#    Responsetest=Response[:,int(train_ratio*T):T,:]
#    [Q,T,M]=np.array(Predictortrain.shape,dtype=int)
#    P=np.array(Responsetrain.shape,dtype=int)[0]
#    Rank=2
#    mu=0.01
#    K=1
#    alpha=0.5
#    print("Point II")
#    listofresponsestrain=[Responsetrain[:,0:100,:]]
#    listofresponsestrain.append(Responsetrain[:,100:200,:])
#    listofpredictorstrain=[Predictortrain[:,0:100,:]]
#    listofpredictorstrain.append(Predictortrain[:,100:200,:])
#    for l in np.array(np.linspace(3,10,8),dtype=int):
#       listofpredictorstrain.append(Predictortrain[:,(l-1)*100:l*100,:])
#       listofresponsestrain.append(Responsetrain[:,(l-1)*100:l*100,:])
#    print("Point III")
#    listofresponsestrain.append(Responsetrain[:,1000:1080,:])
#    listofpredictorstrain.append(Predictortrain[:,1000:1080,:])
#    Methodname="Tuckerfull"
#    Parametertensor,rmselist=CodingAltoProblem.OnlineTensorlearningallblocks(Similaritymatrix,listofresponsestrain,listofpredictorstrain,Rank,P,Q,M,K,mu,alpha,Methodname)
#    rmse=CodingAltoProblem.RMSE(Responsetest,Parametertensor,Predictortest)
#    RMSE.append(rmse)
#    print(rmse)
#    pdb.set_trace()
#print(np.mean(np.array(RMSE)))
pdb.set_trace() 
#Online  method
#RMSE for lag=1
RMSEarray=np.array([2.122,1.814,1.713,2.054,1.796,1.888,1.940,1.789,1.761,1.734])
#RMSE for lag=2
RMSEarray=np.array([2.123,1.814,1.713,2.054,1.796,1.888,1.939,1.789,1.762,1.734])
#RMSE for lag=3
RMSEarray=np.array([2.123,1.814,1.713,2.055,1.796,1.888,1.940,1.790,1.762,1.734])
 
#Tucker2 method
#RMSE for lag=1
RMSEarray=np.array([2.154,1.884,1.725,2.069,1.828,1.931,1.952,1.799,1.793,1.761])
#RMSE for lag=2
RMSEarray=np.array([2.406,1.829,1.728,2.071,1.830,1.907,1.948,1.815,1.781,1.743])
#RMSE for lag=3
RMSEarray=np.array([2.129,1.830,1.728,2.103,1.831,1.898,1.963,1.813,1.769,1.819])

#Tuckerfull method
#RMSE for lag=1
RMSEarray=np.array([2.123,1.815,1.713,2.055,1.796,1.888,1.940,1.789,1.761,1.734])
#RMSE for lag=2
RMSEarray=np.array([2.123,1.814,1.713,2.054,1.796,1.888,1.940,1.790,1.762,1.734])
#RMSE for lag=3
RMSEarray=np.array([2.123,1.814,1.713,2.055,1.796,1.888,1.940,1.789,1.762,1.734])

#Alto method
#RMSE for lag=1
RMSEarray=np.array([2.143,2.945,2.929,2.636,2.572,1.886,2.015,1.797,1.769,1.838])
#RMSE for lag=2
RMSEarray=np.array([2.128,2.927,2.728,2.682,2.728,1.921,1.98,1.787,1.804,1.820])
#RMSE for lag=3
RMSEarray=np.array([2.139,3.304,2.691,2.724,3.199,1.940,1.988,1.778,1.790,1.8637])


