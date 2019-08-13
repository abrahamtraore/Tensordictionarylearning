#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:27:57 2018

@author: Traoreabraham
"""



import sys
sys.path.append("..")
import tensorly as tl
tl.set_backend('numpy')
from tensorly.base import unfold
import numpy as np
np.random.seed(1)
import random
random.seed(1)

from tensorly import tenalg
from tensorly.backend import mxnet_backend


def TensorDataDrawnRandomly(X_set):
    Numberofexamples=len(X_set)
    Randomindices=random.sample(range(0,Numberofexamples), Numberofexamples)
    X_setreshuffled=[]
    for l in Randomindices:
        X_setreshuffled.append(X_set[l])
    return X_setreshuffled

def Operations_listmatrices(listofmatrices,operationnature):
    #This function takes a list of matrices and performs some classical operations on its elements.
    #The variable operationnature specifies the operation performed
    #The matrices are of tensor type
    Res=[]
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
   
    if(operationnature=="Timestranspose"):
       for matrix in listofmatrices:
           element=np.copy(mxnet_backend.to_numpy(matrix))
           Res.append(tl.tensor(np.dot(element,element.T))) #computes A*A.T  
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
               
def Tensor_matrixproduct(X,listoffactors):# The parameters are tensors
    #This function computes the product of a N-order tensor with  N matrices    
    #X is of tensor type as well as the matrices    
    Res=tl.tensor(X)
    mode=-1
    for matrix in listoffactors:
        mode=mode+1
        
        Res=tenalg.mode_dot(Res,matrix,mode)        
    return Res

def ErrorSingle(args):#The parameters are tensors
   #This function computes the square of the fitting error
   error=np.power(tl.norm(args[0][args[3]]-Tensor_matrixproduct(args[1][args[3]],args[2]),2),2)
   return error

def ErrorSet(X_set,G,listoffactors,pool):#The parameters are tensors 
    #This function computes the square of the fitting error for several tensors
    #All the paramters are of tensor type
    L=len(X_set)
    Result=pool.map(ErrorSingle,[[X_set,G,listoffactors,l] for  l in range(L)])    
    return np.array(Result)

def Error(X,G,listoffactors,setting,pool):#The parameters are tensors
    #This function computes the fitting error in batch and online setting
    #All the parameters are of tensor type
    if(setting=="Single"):
        error=np.power(tl.norm(X-Tensor_matrixproduct(G,listoffactors),2),2)
        return error
    if(setting=="MiniBatch"):
        Errorlist=ErrorSet(X,G,listoffactors,pool)
        return np.mean(np.array(Errorlist))
       
def RobustsubspaceLearning_Single(X,Pre_existingprojectionmatrices,Pre_existingenergymatrices,Pre_existingmean,beta,alpha,p):#All parameters are arrays    
    Tensor=tl.tensor(X)
    listoffactors=list(Pre_existingprojectionmatrices)
    listoffactors=Operations_listmatrices(listoffactors,"Tensorize")
    Energymatrices=list(Pre_existingenergymatrices)
    Mean=np.copy(Pre_existingmean)
    N=len(list(Tensor.shape))
    R=Tensor-Tensor_matrixproduct(X,Operations_listmatrices(listoffactors,"Transposetimes"))  
    Weightmatriceslist=[]
    for n in range(N):
        Eigenvalue=np.linalg.eig(Energymatrices[n])[0]
        U=listoffactors[n]
        [I,J]=np.array(U.shape,dtype=int)
        Xn=unfold(Tensor,mode=n) 
        [In,Jn]=np.array(Xn.shape,dtype=int)
        Weightmatrix=np.zeros((In,Jn))               
        Sigma=np.zeros((In,Jn))        
        for i in range(In):
          for j in range(Jn):
            Sigma[i,j]=np.max(np.multiply(np.sqrt(np.abs(Eigenvalue[1:p])),mxnet_backend.to_numpy(U[1:p,i])))       
        k=beta*Sigma  
        if(n==1):
            R=R.T        
        for i in range(In):
           for j in range(Jn):
        
              #Weightmatrix[i,j]=1/(1+np.power(mxnet_backend.to_numpy(R[i,j])/np.maximum(k[i,j],0.001),2)):#This was the initial line
              Weightmatrix[i,j]=1/(1+np.power(R[i,j]/np.maximum(k[i,j],0.001),2))
        Weightmatriceslist.append(Weightmatrix)
    W=np.minimum(Weightmatriceslist[0],Weightmatriceslist[1].T)
    
    WeightTensor=tl.tensor(np.multiply(np.sqrt(mxnet_backend.to_numpy(W)),mxnet_backend.to_numpy(Tensor)))
    Mean=alpha*Mean+(1-alpha)*mxnet_backend.to_numpy(WeightTensor)  
    Projectionmatricesresult=[]   
    Energymatreicesresult=[]    
    for n in range(N):
        Xn=unfold(WeightTensor,mode=n)
        Covariancematrix=np.dot(np.dot(mxnet_backend.to_numpy(listoffactors[n]).T,Energymatrices[n]),mxnet_backend.to_numpy(listoffactors[n]))
        Covariancematrix=alpha*Covariancematrix+(1-alpha)*np.dot(mxnet_backend.to_numpy(Xn),mxnet_backend.to_numpy(Xn).T)
        [Un,diagn,V]=np.linalg.svd(Covariancematrix)
        
        diagn=diagn/np.power(tl.norm(Xn,2),2)
        indices=np.argsort(diagn)
        indices=np.flip(indices,axis=0)

        [J,I]=np.array(listoffactors[n].shape,dtype=int)
        Unew=np.zeros((J,I))
        for j in range(J):
           Unew[j,:]=Un[indices[j],:]
        Sn=np.diag(diagn)     
        Projectionmatricesresult.append(Unew)
        Energymatreicesresult.append(Sn)
    return Projectionmatricesresult,Energymatreicesresult,Mean,WeightTensor     
 
def RobustsubspaceLearning_Set(X_set,Pre_existingprojectionmatrices,Pre_existingenergymatrices,Pre_existingmean,beta,alpha,p,pool):
    Projectionmatricesnew=list(Pre_existingprojectionmatrices)
    Projectionmatricesold=[]
    Energymatricesnew=list(Pre_existingenergymatrices)
    Energymatricesold=[]
    Meannew=Pre_existingmean
    Meanold=np.zeros(Pre_existingmean.shape)
    for t in range(len(X_set)):
        X=tl.tensor(X_set[t])
        Projectionmatricesold=Operations_listmatrices(Projectionmatricesnew,"Tensorize")
        Energymatricesold=Energymatricesnew
        Meanold=Meannew
        Projectionmatricesnew,Energymatreicesnew,Mean,WeightTensor=RobustsubspaceLearning_Single(X,Projectionmatricesold,Energymatricesold,Meanold,beta,alpha,p)
        #pdb.set_trace()
    G_set=[]
    for t in range(len(X_set)):
        X=tl.tensor(X_set[t]) 
        G_set.append(Tensor_matrixproduct(X,Operations_listmatrices(Projectionmatricesnew,"Tensorize")))
    Projectionmatricesresult=Operations_listmatrices(Projectionmatricesnew,"Tensorize")
    
    return Projectionmatricesresult,Energymatricesnew,Mean,WeightTensor,G_set


def RobustsubspaceLearning_SetWithPredefinedEpochs(X_set,Pre_existingprojectionmatrices,Pre_existingenergymatrices,Pre_existingmean,beta,alpha,p,nbepochs,pool):
    
  
    Energymatricesresult=list(Pre_existingenergymatrices)
    Energymatricesold=[]
    
    Projectionmatricesresult=list(Pre_existingprojectionmatrices)
    Projectionmatricesold=[]

    for nepoch in range(nbepochs):
        Projectionmatricesold=Projectionmatricesresult
        Energymatricesold=Energymatricesresult
        np.random.seed(nepoch)
        X_setreshuffled=TensorDataDrawnRandomly(X_set)
        Projectionmatricesresult,Energymatricesreult,Mean,WeightTensor,G_set=RobustsubspaceLearning_Set(X_setreshuffled,Projectionmatricesold,Energymatricesold,Pre_existingmean,beta,alpha,p,pool)     
    return Projectionmatricesresult
