#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:14:19 2018

@author: Traoreabraham
"""


import numpy as np
import scipy
import sys
#sys.path.append("/home/scr/etu/sil821/traorabr/OnlineTensorDictionaryLearning/")
sys.path.append("..")

#from MiscellaneousFunctions.OnlineTensor import CyclicBlocCoordinateTucker_setWithPredefinedEpochs
#from MiscellaneousFunctions.TuckerBatch import TuckerBatch
#from ALTO import ALTO_setWithpredefinedEpochs


import tensorly as tl
from tensorly.backend import mxnet_backend
from tensorly import tenalg
from tensorly.base import unfold
import random

def Nonnegativepart(Elements):#The parameters are arrays
    result=[]
    for element in Elements:
        result.append(np.maximum(element,0))
    return result

def HOSVD(Tensor,Coretensorsize):#The parameter is a tensor
    N=len(Tensor.shape)
    listofmatrices=[]
    for n in range(N):
        U,s,V=np.linalg.svd(mxnet_backend.to_numpy(unfold(Tensor,n)),full_matrices=True)
        A=U[:,0:Coretensorsize[n]]
        listofmatrices.append(A)        
    Coretensor=Tensor_matrixproduct(tl.tensor(Tensor),Operations_listmatrices(Operations_listmatrices(listofmatrices,"Transpose"),"Tensorize"))        
    Coretensor=mxnet_backend.to_numpy(Coretensor)
    return Coretensor,listofmatrices

def pool_init():  
    import gc
    gc.collect()
    
def Tensor_matrixproduct(X,listoffactors):#The parameters are tensors(tensor and matrices)
    
    Res=tl.tensor(np.copy(mxnet_backend.to_numpy(X)))
    
    mode=-1
    for matrix in listoffactors:
        mode=mode+1
        
        Res=tenalg.mode_dot(Res,matrix,mode) 
       
    return Res

def GenerateTensorsGeneral(Numberofexamples,eta,randomseed):
    np.random.seed(randomseed)
    Xtrain=np.random.rand(Numberofexamples,30,40,50)
    Coretensorsize=np.array([Numberofexamples,eta,eta,eta])    
    Greal=np.random.normal(loc=0,scale=1/5,size=Coretensorsize)
    #The line below changes
    #listoffactorsreal=[np.random.normal(loc=0,scale=1/10,size=(30,20)),np.random.normal(loc=0,scale=1/10,size=(40,20)),np.random.normal(loc=0,scale=1/10,size=(50,20))]         
    listoffactorsreal=[np.random.normal(loc=0,scale=1/10,size=(30,eta)),np.random.normal(loc=0,scale=1/10,size=(40,eta)),np.random.normal(loc=0,scale=1/10,size=(50,eta))]             
    for n in range(Numberofexamples):
       Xtrain[n,:,:,:]=mxnet_backend.to_numpy(Tensor_matrixproduct(tl.tensor(Greal[n,:,:,:]),Operations_listmatrices(listoffactorsreal,"Tensorize")))     
    #Xtrain=Xtrain/T.norm(T.tensor(Xtrain),2)
    return Xtrain

#import TuckerBatch
def GenerateTensorsNonnegative(Numberofexamples,eta,randomseed):
    np.random.seed(randomseed)
    Xtrain=np.random.rand(Numberofexamples,30,40,50)
    Coretensorsize=np.array([Numberofexamples,eta,eta,eta])    
    Greal=np.random.normal(loc=0,scale=1/100,size=Coretensorsize)
    #The line below changes
    #listoffactorsreal=[np.random.normal(loc=0,scale=1/10,size=(30,20)),np.random.normal(loc=0,scale=1/10,size=(40,20)),np.random.normal(loc=0,scale=1/10,size=(50,20))]         
    listoffactorsreal=[np.random.normal(loc=0,scale=1/100,size=(30,eta)),np.random.normal(loc=0,scale=1/100,size=(40,eta)),np.random.normal(loc=0,scale=1/100,size=(50,eta))]             
    for n in range(Numberofexamples):
       Xtrain[n,:,:,:]=mxnet_backend.to_numpy(Tensor_matrixproduct(tl.tensor(Greal[n,:,:,:]),Operations_listmatrices(listoffactorsreal,"Tensorize")))     
    Xtrain=np.maximum(Xtrain,0)+np.maximum(np.random.normal(loc=0,scale=1,size=(Numberofexamples,30,40,50)),0)
    return Xtrain



def Operations_listmatrices(listofmatrices,operationnature):#The parameters are tensors
    Res=[]
    if (operationnature=="Turnintoarray"):
        for matrix in listofmatrices:
           element=np.copy(mxnet_backend.to_numpy(matrix))
           Res.append(element)#computes A.T
        return Res
    
    if (operationnature=="Transpose"):
        for matrix in listofmatrices:
           element=np.copy(mxnet_backend.to_numpy(matrix))
           Res.append(tl.tensor(element.T))#computes A.T
        return Res
    
    if(operationnature=="Transposetimes"):
       for matrix in listofmatrices:
           element=np.copy(mxnet_backend.to_numpy(matrix))
         
           Res.append(tl.tensor(np.dot(element.T,element))) #computes A.T*A  
       return Res
   
    if(operationnature=="NormI"):
           for matrix in listofmatrices:
               Res.append(tl.norm(matrix,1))
           return Res
    if(operationnature=="NormII"):
           for matrix in listofmatrices:
               Res.append(np.power(tl.norm(matrix,2),2))
           return Res
       
    if(operationnature=="Tensorize"):
           for matrix in listofmatrices:
               Res.append(tl.tensor(matrix))
           return Res
       
def TensorDataDrawnRandomly(X_set):
    Numberofexamples=len(X_set)
    Randomindices=random.sample(range(0,Numberofexamples), Numberofexamples)
    X_setreshuffled=[]
    for l in Randomindices:
        X_setreshuffled.append(X_set[l])
    return X_setreshuffled

def Transfortensorintosubsequence(Tensor):#the parameter is a numpy array
    result=[]
    L=np.array(Tensor.shape,dtype=int)[0]
    for l in range(L):
        result.append(tl.tensor(Tensor[l,:,:]))
    return result

def Derivativefeatureproblem(Spectrogram,G,A,B,alpha,theta,Nonnegative):#The parameters are tensors
    derivative=-np.dot(mxnet_backend.to_numpy(A).T,mxnet_backend.to_numpy(Spectrogram)-np.dot(np.dot(mxnet_backend.to_numpy(A),mxnet_backend.to_numpy(G)),mxnet_backend.to_numpy(B).T))
    derivative=np.dot(derivative,mxnet_backend.to_numpy(B))
    if(Nonnegative==True):
        derivative=derivative+alpha*theta*np.ones(G.shape)
    return tl.tensor(derivative)
      
def Activationcoeffsingle(args):#The parameters are tensors  
    
    Gnew=tl.tensor(args[1][args[10]])
    Gold=tl.tensor(np.zeros(args[1][args[10]].shape))
    Gresult=tl.tensor(np.zeros(args[1][args[10]].shape))    
    Matrix=np.dot(np.dot(mxnet_backend.to_numpy(args[2]),mxnet_backend.to_numpy(Gnew)),mxnet_backend.to_numpy(args[3]).T)
    error=tl.norm(args[0][args[10]]-tl.tensor(Matrix),2)/tl.norm(args[0][args[10]],2)
    nbiter=0
  
    while(nbiter<args[5]):
        nbiter=nbiter+1
        Gold=Gnew
        derivative=Derivativefeatureproblem(args[0][args[10]],Gold,args[2],args[3],args[7],args[8],args[9])
        
        Gnew=Gold-args[4]*derivative
        if(args[9]==True):
          Gnew=tl.tensor(np.maximum(mxnet_backend.to_numpy(Gnew),0))
        Gresult=Gnew

        
        Matrix=np.dot(np.dot(mxnet_backend.to_numpy(args[2]),mxnet_backend.to_numpy(Gnew)),mxnet_backend.to_numpy(args[3]).T)
        
        error=tl.norm(args[0][args[10]]-tl.tensor(Matrix),2)/tl.norm(args[0][args[10]],2)
        if(error<args[6]):
            break
    return Gresult


#This function is used to turn a tensor into a matrix   
def Transform_tensor_into_featuresmatrix(tensor):
    size=np.array(tensor.shape,dtype=int)
    number_of_samples=size[0]
    result=np.zeros((number_of_samples,size[1]*size[2]))
    for i in range(number_of_samples):
        result[i,:]=np.resize(mxnet_backend.to_numpy(tensor[i,:,:]),np.size(tensor[i,:,:]))
    return result

def ComputeCosineDistance(A,B):
    [N,M]=np.array(A.shape,dtype=int)
    res=[]
    for n in range(N):
        if ((np.linalg.norm(A[n,:])!=0) and (np.linalg.norm(B[n,:])!=0)):
           cosinus=scipy.spatial.distance.cosine(A[n,:],B[n,:])
           #cosinus=scipy.spatial.distance.correlation(A[:,m],B[:,m])
           #print("The value of cosinus is")
           #print(cosinus)
           res.append(cosinus)
    return np.mean(np.array(res))

def ComputeCosineDistanceSet(Listofactorsa,Listoffactorsb):
    N=len(Listofactorsa)
    res=[]
    for n in range(N):
        A=Listofactorsa[n]
        B=Listoffactorsb[n]
        res.append(ComputeCosineDistance(A,B))
    return np.array(res) 
    
   
def CheckSameSubspaces(A,B):#Compare the subspaces generated by the two matrices
    Qa=np.linalg.qr(A)[0]
    Qb=np.linalg.qr(B)[0]
    return Qa,Qb

def CheckSameSubspacesSet(Listofactorsa,Listoffactorsb):
    N=len(Listofactorsa)
    res=[]
    for n in range(N):
        Qa=Listofactorsa[n]
        Qb=Listoffactorsb[n]
        res.append(np.linalg.norm(Qa-Qb))
    return np.array(res)

def Sumallfactors(Listofactorsa):
    N=len(Listofactorsa)
    res=0
    for n in range(N):
        res=res+np.sum(Listofactorsa[n])
    return np.array(res) 