#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 16:46:20 2018

@author: Traoreabraham
"""

import numpy as np
from multiprocessing import Pool
import sys
import pdb
#sys.path.append("/home/scr/etu/sil821/traorabr/OnlineTensorDictionaryLearning/")
sys.path.append("/Users/Traoreabraham/Desktop/OnlineTensorDictionaryLearning/")
from MiscellaneousFunctions.ALTO import ALTO_single
from MiscellaneousFunctions.MethodsTSPen import Tensor_matrixproduct
from MiscellaneousFunctions.MethodsTSPen import Operations_listmatrices
from MiscellaneousFunctions.MethodsTSPen import HOSVD
from MiscellaneousFunctions.OnlineTensor import CyclicBlocCoordinateTucker_setWithPredefinedEpochs
from MiscellaneousFunctions.MethodsTSPen import Operations_listmatrices
sys.path.append("/home/scr/etu/sil821/traorabr/Tensorly/")
import tensorly as tl
from tensorly.backend import mxnet_backend
from tensorly.decomposition import tucker
np.random.seed(2)

def OnlineTensorlearningsingleblock(Choleskysimilaritymatrix,Oldparamtensor,Oldloadingmatrices,ResponsetensorX,PredictortensorZ,alpha,M,K,Coretensorsize):
    #All the operations, i.e. the variable change, the update and ALTO application are performed in the function
    #In this function, we assume that ALTO has already been applied to the former parameter tensor and the loading matrices already recovered
    
    Responsetensor=np.zeros(ResponsetensorX.shape)
    Newparametertensor=np.zeros(Oldparamtensor.shape)

    for m in range(M):
        #The parameter change is performed
        Responsetensor[:,:,m]=np.dot(np.linalg.inv(Choleskysimilaritymatrix),ResponsetensorX[:,:,m])
        #We update the parameter tensor
        Newparametertensor[:,:,m]=(1-alpha)*Oldparamtensor[:,:,m]+alpha*np.dot(Responsetensor[:,:,m],np.linalg.pinv(PredictortensorZ[:,:,m]))
        #print(Newparametertensor)
        print(Responsetensor)
        print(np.linalg.pinv(PredictortensorZ[:,:,m]))
       
    #Core,Listoffactormarices=ALTO_single(Newparametertensor,Coretensorsize,K,Oldloadingmatrices,1)
    
    Setting="Single"
    [I,J,K]=np.array(Newparametertensor.shape,dtype=int)
    R=Coretensorsize[0]
    Pre_existingfactors=[np.random.normal(loc=0,scale=1/2,size=(I,R)),np.random.normal(loc=0,scale=1/2,size=(J,R)),np.random.normal(loc=0,scale=1/2,size=(K,R))]
    Pre_existingG_set=np.random.normal(loc=0,scale=1/2,size=(R,R,R))
    Pre_existingP=[np.random.normal(loc=0,scale=1/2,size=(I,R)),np.random.normal(loc=0,scale=1/2,size=(J,R)),np.random.normal(loc=0,scale=1/2,size=(K,R))]
    Pre_existingQ=[np.random.normal(loc=0,scale=1/2,size=(R,R)),np.random.normal(loc=0,scale=1/2,size=(R,R)),np.random.normal(loc=0,scale=1/2,size=(R,R))]
    Nonnegative=False
    Reprojectornot=True
    Minibatchnumber=[]
    step=np.power(10,-18,dtype=float)
    alpha=np.power(10,2,dtype=float)           #np.power(10,-2,dtype=float)
    theta=np.power(10,-2,dtype=float)
    max_iter=20
    epsilon=np.power(10,-3,dtype=float)
    period=2
    nbepochs=1
    pool=Pool(10)
  
    Core,Listoffactormarices=CyclicBlocCoordinateTucker_setWithPredefinedEpochs([Newparametertensor],Coretensorsize,Pre_existingfactors,[Pre_existingG_set],Pre_existingP,Pre_existingQ,Nonnegative,Reprojectornot,Setting,Minibatchnumber,step,alpha,theta,max_iter,epsilon,period,nbepochs,pool)

    Newparametertensor=Tensor_matrixproduct(tl.tensor(Core[0]),Operations_listmatrices(Listoffactormarices,"Tensorize"))
    
    #Listoffactormarices: tensor type
    #Newparametertensor: tensor type
    return Newparametertensor,Listoffactormarices

 
    
def OnlineTensorlearningallblocks(Similaritymatrix,listresponses,listpredictors,Rank,P,Q,M,K,mu,alpha):
    #listresponses contain the data for two consecutive time samples
    Choleskysimilaritymatrix=np.copy(Similaritymatrix)
    
    Oldparamtensor=np.zeros((P,Q,M))
    Coretensorsize=[Rank,Rank,Rank]
    rmselist=[]
    for m in range(M):
       
       Oldparamtensor[:,:,m]=np.dot(listresponses[0][:,:,m],np.linalg.pinv(listpredictors[0][:,:,m])) 
           
    Core,Newloadingmatrices=HOSVD(tl.tensor(Oldparamtensor),Coretensorsize)    #tucker(tl.tensor(Oldparamtensor),Coretensorsize,init='svd',random_state=1)
      
    Newparametertensor=Tensor_matrixproduct(tl.tensor(Core),Operations_listmatrices(Newloadingmatrices,"Tensorize"))
    Oldloadingmatrices=[] 
    
    for l in range(len(listresponses)-1):
        ResponsetensorX=listresponses[l+1]
        PredictortensorZ=listpredictors[l+1]
        Oldparamtensor=Newparametertensor
        Oldloadingmatrices=Newloadingmatrices
        
        Newparametertensor,Newloadingmatrices=OnlineTensorlearningsingleblock(Choleskysimilaritymatrix,mxnet_backend.to_numpy(Oldparamtensor),Oldloadingmatrices,ResponsetensorX,PredictortensorZ,alpha,M,K,Coretensorsize)
        
        
        rmselist.append(RMSE(ResponsetensorX,mxnet_backend.to_numpy(Newparametertensor),PredictortensorZ))
        
    return mxnet_backend.to_numpy(Newparametertensor),rmselist

def RMSE(Responsetensor,parametertensor,predictortensor):
    error=0
    M=np.array(predictortensor.shape,dtype=int)[2]
    T=np.array(Responsetensor.shape,dtype=int)[1]
    P=np.array(Responsetensor.shape,dtype=int)[0]
    for m in range(M):
          
           error=error+np.power(np.linalg.norm(Responsetensor[:,:,m]-np.dot(parametertensor[:,:,m],predictortensor[:,:,m])),2)
    error=np.sqrt(error/(T*P*M))
    return error
#P=20
#T=30
#M=40
#Q=50
#Responsetensor=np.random.rand(P,T,M)
#parametertensor=np.random.rand(P,Q,M)
#predictortensor=np.random.rand(Q,T,M)
#error=RMSE(Responsetensor,parametertensor,predictortensor)
#pdb.set_trace() 

def series_to_samples(Responsetensor,nLag):
    [P,T,M]=np.array(Responsetensor.shape,dtype=int)
    Q=P*nLag
    nSample=T-nLag
    Predictor=np.zeros((Q,nSample,M))#np.zeros((1,nSample))
    Response=np.zeros((P,nSample,M))#np.zeros((1,nSample))
    for s in range(T-nLag):
        X_s=np.zeros((Q,M))
        for m in range(M):
           tmp=Responsetensor[:,s:s+nLag,m]
           X_s[:,m]=np.resize(tmp,np.size(tmp))
        Predictor[:,s,:]=X_s;
        Y_s=np.zeros((P,M))
        for m in range(M):
            Y_s[:,m]=Responsetensor[:,s+nLag,m]
        Response[:,s,:]=Y_s  
    return Predictor,Response
            
 
from scipy.io.matlab import mio
nom_fichier="foursquare.mat"
fichier = mio.loadmat(nom_fichier) 
Datafoursquarelist=fichier['series_obs']  
Similaritymatrix=fichier['sim_friend']

mon_fichiercholesky="Choleskyfoursquare.mat"
fichiercholesky= mio.loadmat(mon_fichiercholesky)
Similaritymatrix=fichiercholesky['H']

#fichiercholesky=mio.loadmat("Choleskysimilarity.mat")
#Similaritymatrix=fichiercholesky['H']
Datafoursquaretensor=np.zeros((56,1200,15))
for i in range(15):
    Datafoursquaretensor[:,:,i]=Datafoursquarelist[i][0]
nLag=3
Predictor,Response=series_to_samples(Datafoursquaretensor,nLag)  
T=np.array(Predictor.shape,dtype=int)[1]
train_ratio=0.9
Predictortrain=Predictor[:,0:int(train_ratio*T),:]
Responsetrain=Response[:,0:int(train_ratio*T),:]
Predictortest=Predictor[:,int(train_ratio*T):T,:]
Responsetest=Response[:,int(train_ratio*T):T,:]
[Q,T,M]=np.array(Predictortrain.shape,dtype=int)
P=np.array(Responsetrain.shape,dtype=int)[0]
Rank=2
mu=0.01
K=2
alpha=0.5
listofresponsestrain=[Responsetrain[:,0:200,:]]
listofpredictorstrain=[Predictortrain[:,0:200,:]]
for l in np.array(np.linspace(3,9,7),dtype=int):
    listofpredictorstrain.append(Predictortrain[:,(l-1)*100:l*100,:])
    listofresponsestrain.append(Responsetrain[:,(l-1)*100:l*100,:])
listofresponsestrain.append(Responsetrain[:,900:971,:])
listofpredictorstrain.append(Predictortrain[:,900:971,:])

Parametertensor,rmselist=OnlineTensorlearningallblocks(Similaritymatrix,listofresponsestrain,listofpredictorstrain,Rank,P,Q,M,K,mu,alpha)

Storeaddress='/Users/Traoreabraham/Desktop/OnlineTensorDictionaryLearning/Spatiotemporalpredictionproblem/RMSE'+str(nLag)
np.savez_compressed(Storeaddress,rmsevsiteration=rmselist)
rmse=RMSE(Responsetest,Parametertensor,Predictortest)

  
#T=33000 
#M=20
#P=30
#Q=60
#
#Predictors=np.random.normal(loc=0.001,scale=1/100,size=(Q,T,M))
#
##Predictors=Predictors/np.linalg.norm(Predictors)
#Predictorstrain=Predictors[:,0:30000,:]
#Predictorstest=Predictors[:,30000:33000,:]
#listofpredictorstrain=[Predictors[:,0:200,:]]
#for l in np.array(np.linspace(3,300,298),dtype=int):
#    listofpredictorstrain.append(Predictorstrain[:,(l-1)*100:l*100,:])
#    
#Responses=Toyinaltodefinition(Predictors,P,Q,T,M)
#
#Responsestrain=Responses[:,0:30000,:]
#Responsestest=Responses[:,30000:33000,:]
#listofresponsestrain=[Responsestrain[:,0:200,:]]
#for l in np.array(np.linspace(3,300,298),dtype=int):
#    listofresponsestrain.append(Responsestrain[:,(l-1)*100:l*100,:])
#mu=np.power(10,-2,dtype=float)
##from scipy.io.matlab import mio
##nom_fichier="CholeskyToy.mat"
##fichier = mio.loadmat(nom_fichier) 
##Similaritymatrix=fichier['Choleskysimilaritymatrixtoy']/2
#Similaritymatrix=np.eye(P)
#Rank=2
#K=2
#alpha=1/2
#
#Newparametertensor=OnlineTensorlearningallblocks(Similaritymatrix,listofresponsestrain,listofpredictorstrain,Rank,P,Q,M,K,mu,alpha)
#
#rmse=RMSE(Responsestest,Newparametertensor,Predictorstest)
#   
pdb.set_trace() 