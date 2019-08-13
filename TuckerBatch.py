#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:07:43 2018

@author: Traoreabraham
"""
from multiprocessing import Pool
import pdb
import sys
sys.path.append("/home/scr/etu/sil821/traorabr/TensorlyBIB/")
import tensorly as tl
tl.set_backend('mxnet')
import numpy as np
from tensorly.backend import mxnet_backend
from tensorly.base import unfold
from tensorly import tenalg


def Derivativefeatureproblem(Spectrogram,G,A,B):#The parameters are tensors
    derivative=-np.dot(mxnet_backend.to_numpy(A).T,Spectrogram-np.dot(np.dot(mxnet_backend.to_numpy(A),mxnet_backend.to_numpy(G)),mxnet_backend.to_numpy(B).T))
    derivative=np.dot(derivative,mxnet_backend.to_numpy(B))
    return derivative
      
def derivativeDict(X,G,A,listofmatrices,alpha,theta,n):#the parameters are tensors
    
    listoffactors=list(listofmatrices)
    listoffactors[n]=tl.tensor(np.identity(X.shape[n]))
    
    WidehatX=Tensor_matrixproduct(X,Operations_listmatrices(listoffactors,"Transpose"))
    
    listoffactors[n]=tl.tensor(np.identity(G.shape[n]))
    
    B=unfold(Tensor_matrixproduct(G,listoffactors),n) 
    
  
    Result=tl.tensor(np.dot(mxnet_backend.to_numpy(unfold(WidehatX,n)),mxnet_backend.to_numpy(unfold(G,n)).T))-tl.tensor(np.dot(mxnet_backend.to_numpy(A),np.dot(mxnet_backend.to_numpy(B),mxnet_backend.to_numpy(B).T)))+alpha*(1-theta)*A

    return Result

def Factorupdateproblem(X,G,Ainit,listoffactorsmatrices,alpha,theta,n,maxiter,epsilon):
    
    Anew=tl.tensor(Ainit)
    Aold=tl.tensor(np.zeros(Anew.shape))
    Aresult=tl.tensor(np.zeros(Anew.shape))
    error=np.power(tl.norm(X-Tensor_matrixproduct(G,listoffactorsmatrices),2),2)+alpha*(1-theta)*np.power(tl.norm(Anew,2),2)
    
    #previouserror=0
    nbiter=0
    while(nbiter<maxiter):
        nbiter=nbiter+1
        Aold=Anew
        #previouserror=error
        
        Anew=derivativeDict(X,G,Aold,listoffactorsmatrices,alpha,theta,n)
       
        Anew=Anew/tl.norm(Anew,2)
        error=np.power(tl.norm(X-Tensor_matrixproduct(G,listoffactorsmatrices),2),2)#+alpha*(1-theta)*np.power(T.norm(Anew,2),2)
        Aresult=Anew
        
        #if(previouserror-error<epsilon):
        if(np.sqrt(error)/tl.norm(X,2)<epsilon):
           Aresult=Aold
           break
    return Aresult
    
def Proximal_operator(X,step):
    #This function computes the proximal operator of the l1 norm
    #X is of tensor type
    #Res=np.copy(X)
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
      error=np.power(tl.norm(X-Tensor_matrixproduct(G_new,listoffactors),2),2)#+Lambda*T.norm(G_new,1))
         
      nb_iter=0
      error_list=[error]    
      while(nb_iter<=max_iter):
          nb_iter=nb_iter+1
          G_old=G_new
         
          G_new=G_old-step*derivativeCore(X,G_old,listoffactors)
          if(Nonnegative==True):             
             G_new=np.maximum(mxnet_backend.to_numpy(G_old-step*(derivativeCore(X,G_old,listoffactors)))+alpha*theta*np.ones(G_old.shape),0)
             G_new=tl.tensor(G_new)
          if(Nonnegative==False):
             G_new=Proximal_operator(G_new,step)
          error=np.power(tl.norm(X-Tensor_matrixproduct(G_new,listoffactors),2),2)#+Lambda*T.norm(G_new,1)
          G_result=G_new
          error_list.append(error)
          #if(np.abs(previous_error-error)/error<epsilon):
          if(np.sqrt(error)/tl.norm(X,2)<epsilon):
              G_result=G_old
              error_list=error_list[0:len(error_list)-1]
              break
      return G_result,error_list,nb_iter
  
def Tensor_matrixproduct(X,listoffactors):
    #This function computes the product of a N-order tensor with  N matrices    
    #X is of tensor type as well as the matrices    
    #Res=T.tensor(np.copy(mxnet_backend.to_numpy(X)))
    
   
    Res=tl.tensor(X)    
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
           #element=np.copy(mxnet_backend.to_numpy(matrix))
           element=np.copy(mxnet_backend.to_numpy(matrix))
           Res.append(tl.tensor(element.T))#computes A.T
        return Res
    
    if(operationnature=="Transposetimes"):
       for matrix in listofmatrices:
           #element=np.copy(mxnet_backend.to_numpy(matrix))
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
       
        
def TuckerBatchdecomp(X,Coretensorsize,max_iter,listoffactorsinit,Ginit,Nonnegative,Reprojectornot,alpha,theta,step,epsilon):
    N=len(list(X.shape))
    listoffactorsnew=list(listoffactorsinit)   
    listoffactorsnew=Operations_listmatrices(listoffactorsnew,"Tensorize")
    listoffactorsold=[]
    listoffactorsresult=[]
    Gnew=tl.tensor(np.copy(Ginit))
    Gold=tl.tensor(np.zeros(Ginit.shape))
    Gresult=tl.tensor(np.zeros(Ginit.shape))
    error=np.power(tl.norm(tl.tensor(X)-Tensor_matrixproduct(Gnew,listoffactorsnew),2),2)#+alpha*theta*T.norm(Gnew,1)+alpha*(1-theta)*np.sum(Operations_listmatrices(listoffactorsnew[1:N],"NormII"))                    
    nbiter=0
    errorlist=[error]

   
    while (nbiter<max_iter):
        
             print("We are in batch")
             nbiter=nbiter+1
             
             listoffactorsold=listoffactorsnew
             Gold=Gnew
       
             Gnew=Sparse_code(tl.tensor(X),Gold,listoffactorsold,Nonnegative,step,max_iter,alpha,theta,epsilon)[0]
           
             for n in range(N-1):
            
                 Aold=listoffactorsnew[n+1]
       
                 Anew=Factorupdateproblem(tl.tensor(X),Gnew,Aold,listoffactorsnew,alpha,theta,n+1,max_iter,epsilon)
            
                 listoffactorsnew[n+1]=Anew
            
                      
             error=np.power(tl.norm(tl.tensor(X)-Tensor_matrixproduct(Gnew,listoffactorsnew),2),2)#+alpha*theta*T.norm(Gnew,1)+alpha*(1-theta)*np.sum(Operations_listmatrices(listoffactorsnew[1:N],"NormII"))             
  
             errorlist.append(error)        
             listoffactorsresult=listoffactorsold
             Gresult=Gnew
             #if(previouserror-error<epsilon):
             print("The criterion is")
             print(np.sqrt(error)/tl.norm(tl.tensor(X),2))
       
             if(np.sqrt(error)/tl.norm(tl.tensor(X),2)<epsilon):
                 listoffactorsresult=listoffactorsold
                 Gresult=Gold
                 errorlist=errorlist[0:len(errorlist)-1]
                 break
             #print(errorlist)

    if(Reprojectornot==True):
             return Gresult,listoffactorsresult,errorlist,nbiter   
    if(Reprojectornot==False):
             return listoffactorsresult,errorlist,nbiter