#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:12:27 2018

@author: Traoreabraham
"""

import numpy as np
from multiprocessing import Pool
import sys
sys.path.append("..")
from MiscellaneousFunctions.MethodsTSPen import Tensor_matrixproduct
from MiscellaneousFunctions.MethodsTSPen import Operations_listmatrices
from MiscellaneousFunctions.MethodsTSPen import HOSVD
from MiscellaneousFunctions.OnlineTensor import CyclicBlocCoordinateTucker_setWithPredefinedEpochs
from MiscellaneousFunctions.TuckerBatch import TuckerBatch
from MiscellaneousFunctions.TuckerBatch import TuckerBatchfull
import tensorly as tl
from tensorly.backend import mxnet_backend
np.random.seed(1)

def OnlineTensorlearningsingleblock(Choleskysimilaritymatrix,Oldparamtensor,Oldloadingmatrices,ResponsetensorX,PredictortensorZ,alpha,M,K,Coretensorsize,Methodname):
    #All the operations, i.e. the variable change, the update and ALTO application are performed in the function
    #In this function, we assume that ALTO has already been applied to the former parameter tensor and the loading matrices already recovered
    
    Responsetensor=np.zeros(ResponsetensorX.shape)
    Newparametertensor=np.zeros(Oldparamtensor.shape)
    Listoffactormarices=[]
    R=Coretensorsize[0]
    for m in range(M):
        #The parameter change is performed
        Responsetensor[:,:,m]=np.dot(np.linalg.inv(Choleskysimilaritymatrix),ResponsetensorX[:,:,m])
        #We update the parameter tensor
        Newparametertensor[:,:,m]=(1-alpha)*Oldparamtensor[:,:,m]+alpha*np.dot(Responsetensor[:,:,m],np.linalg.pinv(PredictortensorZ[:,:,m]))
       
    if(Methodname=="Online"):
       Setting="Single"
       [I,J,K]=np.array(Newparametertensor.shape,dtype=int)
       
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
       
    if(Methodname=="Tucker"):
        
       Ginit=np.random.normal(loc=0,scale=1/2,size=(R,R,R))
       [I,J,K]=np.array(Newparametertensor.shape,dtype=int)
       Reprojectornot=True
       max_iter=20
       step=np.power(10,-18,dtype=float)
       alpha=np.power(10,2,dtype=float)         
       theta=np.power(10,-2,dtype=float)
       epsilon=np.power(10,-3,dtype=float)
       listoffactorsinit=[np.random.normal(loc=0,scale=1,size=(I,R)),np.random.normal(loc=0,scale=1/2,size=(J,R)),np.random.normal(loc=0,scale=1/2,size=(K,R))]
       X=Newparametertensor
       Nonnegative=False
       Core,Listoffactormarices,errorlist,nbiter=TuckerBatch(X,Coretensorsize,max_iter,listoffactorsinit,Ginit,Nonnegative,Reprojectornot,alpha,theta,step,epsilon)                                           
       Newparametertensor=Tensor_matrixproduct(tl.tensor(Core),Operations_listmatrices(Listoffactormarices,"Tensorize"))
    
    if(Methodname=="Tuckerfull"):
       [I,J,K]=np.array(Newparametertensor.shape,dtype=int)
       Ginit=np.random.normal(loc=0,scale=1/2,size=(I,R,R))
       
       Reprojectornot=True
       max_iter=20
       step=np.power(10,-18,dtype=float)
       alpha=np.power(10,2,dtype=float)         
       theta=np.power(10,-2,dtype=float)
       epsilon=np.power(10,-3,dtype=float)
       listoffactorsinit=[np.eye(I),np.random.normal(loc=0,scale=1/2,size=(J,R)),np.random.normal(loc=0,scale=1/2,size=(K,R))]
       X=Newparametertensor
       Nonnegative=False
       Core,Listoffactormarices,errorlist,nbiter=TuckerBatchfull(X,Coretensorsize,max_iter,listoffactorsinit,Ginit,Nonnegative,Reprojectornot,alpha,theta,step,epsilon)                                           
       Newparametertensor=Tensor_matrixproduct(tl.tensor(Core),Operations_listmatrices(Listoffactormarices,"Tensorize"))
    
    #Listoffactormarices: tensor type
    #Newparametertensor: tensor type
    return Newparametertensor,Listoffactormarices

 
    
def OnlineTensorlearningallblocks(Similaritymatrix,listresponses,listpredictors,Rank,P,Q,M,K,mu,alpha,Methodname):
    #listresponses contain the data for two consecutive time samples
    Choleskysimilaritymatrix=np.copy(Similaritymatrix)
    
    Oldparamtensor=np.zeros((P,Q,M))
    Coretensorsize=[Rank,Rank,Rank]
    rmselist=[]
    for m in range(M):
       
       #pdb.set_trace()
       Oldparamtensor[:,:,m]=np.dot(listresponses[0][:,:,m],np.linalg.pinv(listpredictors[0][:,:,m])) 
           
    Core,Newloadingmatrices=HOSVD(tl.tensor(Oldparamtensor),Coretensorsize)    #tucker(tl.tensor(Oldparamtensor),Coretensorsize,init='svd',random_state=1)
      
    Newparametertensor=Tensor_matrixproduct(tl.tensor(Core),Operations_listmatrices(Newloadingmatrices,"Tensorize"))
    Oldloadingmatrices=[] 
    
    for l in range(len(listresponses)-1):
        ResponsetensorX=listresponses[l+1]
        PredictortensorZ=listpredictors[l+1]
        Oldparamtensor=Newparametertensor
        Oldloadingmatrices=Newloadingmatrices
        print("The block number is")
        print(l)
        Newparametertensor,Newloadingmatrices=OnlineTensorlearningsingleblock(Choleskysimilaritymatrix,mxnet_backend.to_numpy(Oldparamtensor),Oldloadingmatrices,ResponsetensorX,PredictortensorZ,alpha,M,K,Coretensorsize,Methodname)        
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
            
