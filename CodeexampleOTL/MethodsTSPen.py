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
backendchoice='numpy'
tl.set_backend(backendchoice)
from tensorly.backend import mxnet_backend
from tensorly import tenalg
from tensorly.base import unfold
import random

import pdb




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

def Retraction(Point,Parameter,backendchoice):
    M=Point+Parameter
    if(backendchoice=='numpy'):
       Q,R=np.linalg.qr(M)
       return tl.tensor(Q)
    if(backendchoice=='mxnet'):
       Q,R=np.linalg.qr(mxnet_backend.to_numpy(M))
       return tl.tensor(Q)

def Projectiononstiefelmanifold(Point,Parameter):
    p=np.array(Parameter.shape,dtype=int)[0]    
    I=tl.tensor(np.identity(p))
    Result=tl.backend.dot(I-tl.backend.dot(Point,Point.T),Parameter)
    #P=tl.dot(Point.T,Parameter)
    #P=Skewmatrix(P)
    Result=Result+tl.backend.dot(Point,Skewmatrix(tl.backend.dot(Point.T,Parameter)))
    return Result

def Retraction(Point,Parameter,backendchoice):
    M=Point+Parameter
    if(backendchoice=='numpy'):
       Q,R=np.linalg.qr(M)
    if(backendchoice=='mxnet'):
       Q,R=np.linalg.qr(mxnet_backend.to_numpy(M))
    return tl.tensor(Q)

#Point=np.random.rand(20,30)
#Parameter=np.random.rand(20,30)
#Q=Retraction(Point,Parameter,backendchoice)
#pdb.set_trace()

def Skewmatrix(M):
  
    Result=(M-M.T)/2
    
    return Result

#M=tl.tensor(np.random.rand(20,20))
#Result=Skewmatrix(M)  
#pdb.set_trace()  
      
def Projectiononstiefelmanifold(Point, Parameter):
    p=np.array(Parameter.shape,dtype=int)[0]    
    I=tl.tensor(np.identity(p))
    Result=tl.dot(I+tl.dot(Point,Point.T),Parameter)
    #P=tl.dot(Point.T,Parameter)
    #P=Skewmatrix(P)
    Result=Result+tl.dot(Point,Skewmatrix(tl.dot(Point.T,Parameter)))
    return Result

#Point=tl.tensor(np.random.rand(20,30))
#Parameter=tl.tensor(np.random.rand(20,30))
#Result=Projectiononstiefelmanifold(Point, Parameter)
#pdb.set_trace()



def subsample(listofobjectivevalues,sampleperiod):
    L=len(listofobjectivevalues)
    listofobjectivevaluessampled=[]
    for i in range(int(L/sampleperiod)):
        listofobjectivevaluessampled.append(listofobjectivevalues[sampleperiod*i])
    return np.array(listofobjectivevaluessampled)
        

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
    Coretensor=Tensor_matrixproduct(tl.tensor(Tensor),Operations_listmatrices(listofmatrices,"Transpose"))        
    #Coretensor=mxnet_backend.to_numpy(Coretensor)
    
    #Coretensor=Tensor_matrixproduct(Tensor,listofmatrices)        
    Coretensor=tl.tensor(Coretensor)
    return Coretensor,listofmatrices

def pool_init():  
    import gc
    gc.collect()
    
def Tensor_matrixproduct(X,listoffactors):#The parameters are tensors(tensor and matrices)
    
    #Res=tl.tensor(np.copy(mxnet_backend.to_numpy(X)))
    Res=tl.tensor(X)
    
    mode=-1
   
    for matrix in listoffactors:
        mode=mode+1
        
        Res=tenalg.mode_dot(Res,matrix,mode) 
       
    return Res


def Elementapproximation(Activationcoefflist,listoffactors):
    Approximatedelements=[]
    for l in range(len(Activationcoefflist)):        
        Approximatedelements.append(Tensor_matrixproduct(Activationcoefflist[l],listoffactors))       
    return Approximatedelements

def GenerateTensorsGeneral(I,J,K,Numberofexamples,eta,randomseed):
    np.random.seed(randomseed)
    Xtrain=tl.tensor(np.random.rand(Numberofexamples,I,J,K))
    Coretensorsize=np.array([Numberofexamples,eta,eta,eta])    
    Greal=tl.tensor(np.random.normal(loc=0,scale=1/5,size=Coretensorsize))
    #The line below changes
    #listoffactorsreal=[np.random.normal(loc=0,scale=1/10,size=(30,20)),np.random.normal(loc=0,scale=1/10,size=(40,20)),np.random.normal(loc=0,scale=1/10,size=(50,20))]         
    listoffactorsreal=[tl.tensor(np.random.normal(loc=0,scale=1/10,size=(I,eta))),tl.tensor(np.random.normal(loc=0,scale=1/10,size=(J,eta))),tl.tensor(np.random.normal(loc=0,scale=1/10,size=(K,eta)))]             
    for n in range(Numberofexamples):
       #Xtrain[n,:,:,:]=mxnet_backend.to_numpy(Tensor_matrixproduct(tl.tensor(Greal[n,:,:,:]),Operations_listmatrices(listoffactorsreal,"Tensorize")))     
        
       Xtrain[n,:,:,:]=Tensor_matrixproduct(Greal[n,:,:,:],listoffactorsreal)    
    
    return Xtrain

#import TuckerBatch
def GenerateTensorsNonnegative(Numberofexamples,eta,randomseed):
    np.random.seed(randomseed)
    Xtrain=tl.tensor(np.random.rand(Numberofexamples,30,40,50))
    Coretensorsize=np.array([Numberofexamples,eta,eta,eta])    
    Greal=tl.tensor(np.random.normal(loc=0,scale=1/100,size=Coretensorsize))
    #The line below changes
    #listoffactorsreal=[np.random.normal(loc=0,scale=1/10,size=(30,20)),np.random.normal(loc=0,scale=1/10,size=(40,20)),np.random.normal(loc=0,scale=1/10,size=(50,20))]         
    listoffactorsreal=[tl.tensor(np.random.normal(loc=0,scale=1/100,size=(30,eta))),tl.tensor(np.random.normal(loc=0,scale=1/100,size=(40,eta))),tl.tensor(np.random.normal(loc=0,scale=1/100,size=(50,eta)))]             
    for n in range(Numberofexamples):
       Xtrain[n,:,:,:]=Tensor_matrixproduct(tl.tensor(Greal[n,:,:,:]),listoffactorsreal)  
    Xtrain=tl.backend.maximum(Xtrain,0)+tl.backend.maximum(tl.tensornp.random.normal(loc=0,scale=1,size=(Numberofexamples,30,40,50)),0)
    return Xtrain



#def Operations_listmatrices(listofmatrices,operationnature):#The parameters are tensors
#    Res=[]
#    if (operationnature=="Turnintoarray"):
#        for matrix in listofmatrices:
#           element=np.copy(mxnet_backend.to_numpy(matrix))
#           Res.append(element)#computes A.T
#        return Res
#    
#    if (operationnature=="Transpose"):
#        for matrix in listofmatrices:
#           element=np.copy(mxnet_backend.to_numpy(matrix))
#           Res.append(tl.tensor(element.T))#computes A.T
#        return Res
#    
#    if(operationnature=="Transposetimes"):
#       for matrix in listofmatrices:
#           element=np.copy(mxnet_backend.to_numpy(matrix))
#         
#           Res.append(tl.tensor(np.dot(element.T,element))) #computes A.T*A  
#       return Res
#   
#    if(operationnature=="NormI"):
#           for matrix in listofmatrices:
#               Res.append(tl.norm(matrix,1))
#           return Res
#    if(operationnature=="NormII"):
#           for matrix in listofmatrices:
#               Res.append(np.power(tl.norm(matrix,2),2))
#           return Res
#       
#    if(operationnature=="Tensorize"):
#           for matrix in listofmatrices:
#               Res.append(tl.tensor(matrix))
#           return Res
       
def Operations_listmatrices(listofmatrices,operationnature):#The parameters are tensors
    Res=[]   
    if (operationnature=="Arrayconversion"):
        for matrix in listofmatrices:
           element=tl.tensor(matrix)
           Res.append(mxnet_backend.to_numpy(element))  
        return Res

    if (operationnature=="Transpose"):
        for matrix in listofmatrices:
           element=tl.tensor(matrix)
           Res.append(element.T)#computes A.T
        return Res
    
    if(operationnature=="Transposetimes"):
       for matrix in listofmatrices:
           element=tl.tensor(matrix)
           Matrix=tl.backend.dot(element.T,element)
           Res.append(Matrix) #computes A.T*A  
       return Res
   
    if(operationnature=="NormI"):
           for matrix in listofmatrices:
               Res.append(tl.norm(matrix,1))
           return Res
    if(operationnature=="NormII"):
           for matrix in listofmatrices:
               Res.append(np.power(tl.norm(matrix,2),2))
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
    #derivative=-np.dot(mxnet_backend.to_numpy(A).T,mxnet_backend.to_numpy(Spectrogram)-np.dot(np.dot(mxnet_backend.to_numpy(A),mxnet_backend.to_numpy(G)),mxnet_backend.to_numpy(B).T))
    #derivative=np.dot(derivative,mxnet_backend.to_numpy(B))

    derivative=-tl.backend.dot(A.T,Spectrogram)-np.dot(np.dot(A,G),B.T)
    derivative=tl.backend.dot(derivative,B)    
    if(Nonnegative==True):
        derivative=derivative+alpha*theta*tl.tensor(np.ones(G.shape))
    return tl.tensor(derivative)
      
def Activationcoeffsingle(args):#The parameters are tensors  
    
    Gnew=tl.tensor(args[1][args[10]])
    Gold=tl.tensor(np.zeros(args[1][args[10]].shape))
    Gresult=tl.tensor(np.zeros(args[1][args[10]].shape))    
    #Matrix=np.dot(np.dot(mxnet_backend.to_numpy(args[2]),mxnet_backend.to_numpy(Gnew)),mxnet_backend.to_numpy(args[3]).T)
    
    Matrix=tl.backend.dot(tl.backend.dot(args[2],Gnew),args[3].T)    
    error=tl.norm(args[0][args[10]]-tl.tensor(Matrix),2)/tl.norm(args[0][args[10]],2)
    nbiter=0
  
    while(nbiter<args[5]):
        nbiter=nbiter+1
        Gold=Gnew
        derivative=Derivativefeatureproblem(args[0][args[10]],Gold,args[2],args[3],args[7],args[8],args[9])
        
        Gnew=Gold-args[4]*derivative
        if(args[9]==True):
          #Gnew=tl.tensor(np.maximum(mxnet_backend.to_numpy(Gnew),0))
          Gnew=tl.tensor(tl.backend.maximum(Gnew,0))
          
        Gresult=Gnew

        
        Matrix=tl.backend.dot(tl.backend.dot(args[2],Gnew),args[3].T)
        
        error=tl.norm(args[0][args[10]]-Matrix,2)/tl.norm(args[0][args[10]],2)
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


#tl.set_backend('mxnet')
#A=tl.tensor(np.array([[1,2],[3,4]]))
#print(tl.backend.reshape(A,A.size))
#
#
#A=np.array([[1,2],[3,4]])
#print(np.reshape(A,A.size))
#
#
#print(np.resize(A,A.size))
#
#pdb.set_trace()


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


#def Proximal_operator(X,step):
#    #This function computes the proximal operator of the l1 norm
#    #X is of tensor type
#    #Res=np.copy(X)
#    Res=np.copy(mxnet_backend.to_numpy(X))
#    Res=np.sign(Res)*np.maximum(np.abs(Res)-step,0)
#    return tl.tensor(Res)

def Proximal_operator(X,step):
    #This function computes the proximal operator of the l1 norm
    #X is of tensor type
    #Res=np.copy(X)
    Res=tl.tensor(X)
    Res=tl.backend.sign(Res)*tl.backend.maximum(tl.backend.abs(Res)-step,0)
    return tl.tensor(Res)

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
       
  
       Gtemp=pool.map(Sparse_codingSingle,[[X,G,listoffactors,Nonnegative,step,max_iter,alpha,theta,epsilon,l] for l in range(L)])
  
       G_result=[]
  
       for Goutput in Gtemp:
          G_result.append(Goutput)
       return G_result




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

def Split_into_two_subsets(X_set,trainratio):
    nbtrainsamples=int(trainratio*len(X_set))
    Xtrainset=X_set[0:nbtrainsamples]
    Xtestset=X_set[nbtrainsamples:len(X_set)]
    return Xtrainset,Xtestset

def Error(X,G,listoffactors,setting,pool):
    #This function computes the fitting error in batch and online setting
    #All the parameters are of tensor type
    if(setting=="Single"):
        error=np.power(tl.norm(X-Tensor_matrixproduct(G,listoffactors),2),2)
        return error
    if(setting=="MiniBatch"):
        Errorlist=ErrorSet(X,G,listoffactors,pool)
        return np.sum(np.array(Errorlist))   

def Compute_test_error(X_test,G_init,listoffactors,Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool):#The parameters are tensors
    G_test=Sparse_coding(X_test,G_init,listoffactors,Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool)
    Test_error=Error(X_test,G_test,listoffactors,Setting,pool)
    return Test_error


def CheckSubspace(listoffactors1,listoffactors2):
    N=len(listoffactors1)
    result=np.zeros(N)
    for n in range(N):
        Q1=listoffactors1[n]
        Q2=listoffactors2[n]
        result[n]=np.linalg.norm(np.dot(Q1,Q1.T)-np.dot(Q2,Q2.T))
    return result


def Mean_relative_errorsingle(args):
    error=np.power(tl.norm(args[0][args[3]]-Tensor_matrixproduct(args[1][args[3]],args[2]),2),2)
    error=error/np.power(tl.norm(args[0][args[3]],2),2)
    return error

def Mean_relative_error(X,G,listoffactors,setting,pool):
    if(setting=="Single"):
        return np.power(tl.norm(X-Tensor_matrixproduct(G,listoffactors),2),2)/np.power(tl.norm(X,2),2)
    if(setting=="MiniBatch"):
        Mean_errorslist=pool.map(Mean_relative_errorsingle,[[X,G,listoffactors,l] for l in range(len(X))])
        return np.mean(np.array(Mean_errorslist))

def derivativeDict(X,G,A,listofmatrices,alpha,theta,n):#the parameters are tensors
    
    listoffactors=list(listofmatrices)
    listoffactors[n]=tl.tensor(np.identity(X.shape[n]))
    
    WidehatX=Tensor_matrixproduct(X,Operations_listmatrices(listoffactors,"Transpose"))
    
    listoffactors[n]=tl.tensor(np.identity(G.shape[n]))
    
    B=unfold(Tensor_matrixproduct(G,listoffactors),n) 
    
  
    Result=tl.tensor(np.dot(mxnet_backend.to_numpy(unfold(WidehatX,n)),mxnet_backend.to_numpy(unfold(G,n)).T))-tl.tensor(np.dot(mxnet_backend.to_numpy(A),np.dot(mxnet_backend.to_numpy(B),mxnet_backend.to_numpy(B).T)))+alpha*(1-theta)*A

    #Res1=np.dot(A,np.dot(B,B.T))
    #Res2=alpha*(1-theta)*A
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

   
def Augment(A,K,sigma):#the parameters are tensors
    [nrows,ncols]=np.array(A.shape,dtype=int)
    B=tl.tensor(np.zeros((nrows,ncols+K)))
    #B[:,0:ncols]=mxnet_backend.to_numpy(A)
    B[:,0:ncols]=A
    B[:,ncols:ncols+K]=tl.tensor(np.random.normal(loc=0,scale=sigma,size=(nrows,K)))
    B=Gramschmidt(B)
    return B

def Augmentlist(Listofmatrices,K,sigma):   
    result=[]    
    #for matrix in Listofmatrices:
    for n in range(len(Listofmatrices)):
        matrix=Listofmatrices[n]
        result.append(Augment(matrix,K,sigma))        
    return result    
#def Augment(A,K,sigma):#the parameters are tensors
#    [nrows,ncols]=np.array(A.shape,dtype=int)
#    B=tl.tensor(np.zeros((nrows,ncols+1)))
#    
#    B[:,0:ncols]=A
#    I=np.array(A.shape,dtype=int)[0]
#    tmp=np.random.normal(loc=0,scale=1,size=(I,1))
#    print(tmp.shape)
#    tmp=tmp-np.dot(np.dot(mxnet_backend.to_numpy(A),mxnet_backend.to_numpy(A).T),tmp)
#    tmp=tmp/np.linalg.norm(tmp)
#    
#    B[:,ncols:ncols+1]=tmp
#    return B
    