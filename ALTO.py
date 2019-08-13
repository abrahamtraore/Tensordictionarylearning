#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 20:03:01 2018

@author: Traoreabraham
"""


from multiprocessing import Pool
import sys
sys.path.append("/home/scr/etu/sil821/traorabr/Tensorly/")

import tensorly as tl
tl.set_backend('mxnet')
from tensorly.backend import mxnet_backend
#from tensorly.base import unfold
import numpy as np
#from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
#from math import ceil
import random
from tensorly import tenalg
#from tensorly.backend import numpy_backend

np.random.seed(2)

def TensorDataDrawnRandomly(X_set):
    Numberofexamples=len(X_set)
    Randomindices=random.sample(range(0,Numberofexamples), Numberofexamples)
    X_setreshuffled=[]
    for l in Randomindices:
        X_setreshuffled.append(X_set[l])
    return X_setreshuffled

def Proximal_operator(X,step):
    #This function computes the proximal operator of the l1 norm
    #X is of tensor type
    #Res=np.copy(mxnet_backend.to_numpy(X))
    Res=np.copy(mxnet_backend.to_numpy(X))
    Res=np.sign(Res)*np.maximum(np.abs(Res)-step,0)
    return tl.tensor(Res)

def derivativeCore(X,G,listofmatrices):
    #This function computes the derivative of the differentiable part of the objective function with respect to G
    #All the parameters are of tensor type    
    #Firstterm=T.tensor(np.copy(mxnet_backend.to_numpy(X)))
    Firstterm=tl.tensor(X)
    Firstterm=Tensor_matrixproduct(Firstterm,Operations_listmatrices(listofmatrices,"Transpose"))        
    #Secondterm=T.tensor(np.copy(mxnet_backend.to_numpy(G)))
    Secondterm=tl.tensor(G)
    Secondterm=Tensor_matrixproduct(Secondterm,Operations_listmatrices(listofmatrices,"Transposetimes"))           
    Res=Firstterm-Secondterm
    return Res

def Sparse_code(X,G_init,listoffactors,Nonnegative,step,max_iter,alpha,theta,epsilon):#The parameters are tensors
      #This function is used to perform the sparse coding step
      #All the tensor and parameters are of tensor type
      #G_new=T.tensor(np.copy(mxnet_backend.to_numpy(G_init)))
      G_new=tl.tensor(G_init)
      G_old=tl.tensor(np.zeros(G_new.shape))
      G_result=tl.tensor(np.zeros(G_new.shape))
      Lambda=alpha*theta      
      error=np.power(tl.norm(X-Tensor_matrixproduct(G_new,listoffactors),2),2)+Lambda*tl.norm(G_new,1)
      previous_error=0
      nb_iter=0
      error_list=[error]    
      while(nb_iter<=max_iter):
          nb_iter=nb_iter+1
          previous_error=error
          G_old=G_new
          G_new=G_old-step*derivativeCore(X,G_old,listoffactors)
          if(Nonnegative==True):
             G_new=np.maximum(G_old-step*(derivativeCore(X,G_old,listoffactors)+alpha*theta*np.ones(G_old.shape)),0)
          if(Nonnegative==False):
             G_new=Proximal_operator(G_new,step)
          error=np.power(tl.norm(X-Tensor_matrixproduct(G_new,listoffactors),2),2)+Lambda*tl.norm(G_new,1)
          G_result=G_new
          error_list.append(error)
          if(np.abs(previous_error-error)/error<epsilon):
              G_result=G_old
              error_list=error_list[0:len(error_list)-1]
              break
      return G_result,error_list,nb_iter
  

def Sparse_codingSingle(args):
        #This function is used to infer the activation coefficients for a single tensor
        #All the parameters are of tensor type
        G_result,error_list,nb_iter=Sparse_code(args[0][args[9]],args[1][args[9]],args[2],args[3],args[4],args[5],args[6],args[7],args[8])
        return G_result
     


def Sparse_coding(X,G,listoffactors,Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool):#The parameters are tensors   
     #This function is used to infer the activation coefficients for eiter a single or a sequence of tensors
     #All the tensors are of tensor type
     if(Setting=="Single"):
        G_result,error_list,nb_iter=Sparse_code(X,G,listoffactors,Nonnegative,step,max_iter,alpha,theta,epsilon)
        return G_result
     if(Setting=="MiniBatch"):
       L=len(X)
       G=pool.map(Sparse_codingSingle,[[X,G,listoffactors,Nonnegative,step,max_iter,alpha,theta,epsilon,l] for l in range (L)])
       G_result=[]
       for Goutput in G:
          G_result.append(Goutput)
       return G_result
   
    
def Gramschmidt(A):
    [m,n]=np.array(A.shape,dtype=int)
    Q=tl.tensor(np.zeros((m,n)))
    R=tl.tensor(np.zeros((n,n)))
    for j in range(n):
        v=A[:,j]
        for i in range(j):
            R[i,j]=np.dot(mxnet_backend.to_numpy(Q[:,i]).T,mxnet_backend.to_numpy(A[:,j]))
            v=v-R[i,j]*Q[:,i]
 
        R[j,j]=tl.norm(v,2)
        Q[:,j]=v/R[j,j]
    return Q


def Tensor_matrixproduct(X,listoffactors):
    #This function computes the product of a N-order tensor with  N matrices    
    #X is of tensor type as well as the matrices    
    Res=tl.tensor(np.copy(mxnet_backend.to_numpy(X)))
    mode=-1
    for matrix in listoffactors:
        mode=mode+1
        
        Res=tenalg.mode_dot(Res,matrix,mode)        
    return Res

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
   
    if(operationnature=="NormI"):
           for matrix in listofmatrices:
               Res.append(tl.norm(tl.tensor(matrix),1))
           return Res
    if(operationnature=="NormII"):
           for matrix in listofmatrices:
               Res.append(np.power(tl.norm(tl.tensor(matrix),2),2))
           return Res       
    if(operationnature=="Tensorize"):
           for matrix in listofmatrices:

               Res.append(tl.tensor(matrix))
           return Res
       
def Augment(A,K,sigma):#the parameters are tensors
    [nrows,ncols]=np.array(A.shape,dtype=int)
    B=tl.tensor(np.zeros((nrows,ncols+K)))
    #B[:,0:ncols]=mxnet_backend.to_numpy(A)
    B[:,0:ncols]=A
    B[:,ncols:ncols+K]=np.random.normal(loc=0,scale=sigma,size=(nrows,K))
    B=Gramschmidt(B)
    return B

#A=T.tensor(np.ones((20,30)))
#K=10
#result=Augment(A,K)

def Augmentlist(Listofmatrices,K,sigma):   
    result=[]    
    for matrix in Listofmatrices:        
        result.append(Augment(matrix,K,sigma))        
    return result
#Listofmatrices=[np.random.normal(loc=0,scale=1,size=(20,30)),np.random.normal(loc=0,scale=1,size=(25,35)),np.random.normal(loc=0,scale=1,size=(30,40))]
#K=10
#result=Augmentlist(Listofmatrices,K)

def ALTO_single(X,Coretensorsize,K,Pre_existingfactors,sigma):#All the parameters are tensors
    ListoffactorsU=list(Pre_existingfactors)
    ListoffactorsV=Augmentlist(ListoffactorsU,K,sigma)
    Stilde=Tensor_matrixproduct(X, Operations_listmatrices(ListoffactorsV,"Transpose"))
    core,factors=tucker(Stilde,Coretensorsize,init='random',random_state=1)
    Listoffactorsresult=[]
    for i in range(len(factors)):
        #Listoffactorsresult.append(np.dot(mxnet_backend.to_numpy(ListoffactorsV[i]),factors[i]))
#        print(ListoffactorsV[i].shape)
#        print(factors[i].shape)
#        pdb.set_trace()
        Listoffactorsresult.append(np.dot(mxnet_backend.to_numpy(ListoffactorsV[i]),mxnet_backend.to_numpy(factors[i])))
    Listoffactorsresult=Operations_listmatrices(Listoffactorsresult,"Tensorize")
    return core,Listoffactorsresult
#X=T.tensor(np.random.normal(loc=0,scale=1,size=(30,40,50)))
#Coretensorsize=[25,35,45]
#K=1
#Pre_existingfactors=[T.tensor(np.random.normal(loc=0,scale=1,size=(30,25))),T.tensor(np.random.normal(loc=0,scale=1,size=(40,35))),T.tensor(np.random.normal(loc=0,scale=1,size=(50,45)))]
#core,Listoffactorsresult=ALTO_single(X,Coretensorsize,K,Pre_existingfactors)

def ErrorSingle(args):
   #This function computes the square of the fitting error
   error=np.power(tl.norm(args[0][args[3]]-Tensor_matrixproduct(args[1][args[3]],args[2]),2),2)
   return error

def ErrorSet(X_set,G,listoffactors,pool): 
    #This function computes the square of the fitting error for several tensors
    #All the paramters are of tensor type
    L=len(X_set)
    Result=pool.map(ErrorSingle,[[X_set,G,listoffactors,l] for  l in range(L)])    
    return np.array(Result)

def Error(X,G,listoffactors,setting,pool):
    #This function computes the fitting error in batch and online setting
    #All the parameters are of tensor type
    if(setting=="Single"):
        error=np.power(tl.norm(X-Tensor_matrixproduct(G,listoffactors),2),2)
        return error
    if(setting=="MiniBatch"):
        Errorlist=ErrorSet(X,G,listoffactors,pool)
        return np.mean(np.array(Errorlist))
    
def ALTO_set(X_set,Coretensorsize,Pre_existingfactors,K,pool,sigma):
    Listoffactorsnew=list(Pre_existingfactors)
    Listoffactorsold=[]
    G_set=[]
    for t in range(len(X_set)):
        X=tl.tensor(X_set[t])       
        Listoffactorsold=Listoffactorsnew
        core,Listoffactorsnew=ALTO_single(X,Coretensorsize,K,Listoffactorsold,sigma)
        G_set.append(core)   
    return core,Listoffactorsnew


def ALTO_setWithpredefinedEpochs(X_set,Coretensorsize,Pre_existingfactors,K,pool,sigma,nbepochs):
    listoffactorsresult=list(Pre_existingfactors)
    listoffactorsold=list(Pre_existingfactors)
    for n in range(nbepochs):
        X_setreshuffled=TensorDataDrawnRandomly(X_set)
        listoffactorsold=listoffactorsresult
        core,listoffactorsresult=ALTO_set(X_setreshuffled,Coretensorsize,listoffactorsold,K,pool,sigma)
    return core,listoffactorsresult

    
    
    
    