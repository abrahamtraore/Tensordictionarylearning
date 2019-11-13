#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:13:48 2018

@author: Traoreabraham
"""
from multiprocessing import Pool
import sys
sys.path.append("/home/scr/etu/sil821/traorabr/Tensorly/")
import tensorly as tl
tl.set_backend('mxnet')
from tensorly.base import unfold
#sys.path.append("/home/scr/etu/sil821/traorabr/Numpy/")
import numpy as np
np.random.seed(2)
import random
#import pdb
from tensorly import tenalg
#from tensorly.backend import numpy_backend
from tensorly.backend import mxnet_backend
import pdb
np.random.seed(2)


#Numberofexamples=5
#A=random.sample(range(0,Numberofexamples), Numberofexamples)
#print(A)
#print(len(A))
#pdb.set_trace()

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
    #Res=np.copy(X)
    Res=np.copy(tl.backend.to_numpy(X))
    Res=np.sign(Res)*np.maximum(np.abs(Res)-step,0)
    return tl.tensor(Res)

#X=tl.tensor(np.random.rand(20,20,10))
#Lambda=0.5
#B=Proximal_operator(X,Lambda)
#pdb.set_trace()

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

#X=tl.tensor(np.random.rand(20,15,30))
#Xavant=X
#listoffactors=[tl.tensor(np.random.rand(15,20)),tl.tensor(np.random.rand(20,15)) ,tl.tensor(np.random.rand(25,30))]
#B=Tensor_matrixproduct(X,listoffactors)
#Xapp=X
#print(np.sum(mxnet_backend.to_numpy(Xavant)==mxnet_backend.to_numpy(Xapp)))
#pdb.set_trace()
    
#def Error(X,G,listoffactors,setting):#All the parameters are tensors
#    error=0
#    if(setting=="Single"):          
#          error=np.power(T.norm(X-Tensor_matrixproduct(G,listoffactors),2),2)
#          return error
#    if(setting=="MiniBatch"):
#       rho=len(X)
#       for r in range(rho):
#          error=error+np.power(T.norm(X[r]-Tensor_matrixproduct(G[r],listoffactors),2),2)
#       return error

#X=T.tensor(np.random.normal(loc=0,scale=1,size=(20,30,40)))
#G=T.tensor(np.random.normal(loc=0,scale=1,size=(15,25,35)))
#listoffactors=[np.random.normal(loc=0,scale=1,size=(20,15)),np.random.normal(loc=0,scale=1,size=(30,25)),np.random.normal(loc=0,scale=1,size=(40,35))]
#setting="Single"
#error=Error(X,G,listoffactors,setting)
       
#X=[T.tensor(np.random.normal(loc=0,scale=1,size=(20,30,40))),T.tensor(np.random.normal(loc=0,scale=1,size=(20,30,40))),T.tensor(np.random.normal(loc=0,scale=1,size=(20,30,40)))]
#G=[T.tensor(np.random.normal(loc=0,scale=1,size=(15,25,35))),T.tensor(np.random.normal(loc=0,scale=1,size=(15,25,35))),T.tensor(np.random.normal(loc=0,scale=1,size=(15,25,35)))]
#listoffactors=[np.random.normal(loc=0,scale=1,size=(20,15)),np.random.normal(loc=0,scale=1,size=(30,25)),np.random.normal(loc=0,scale=1,size=(40,35))]
#setting="Minibatch"
#Error(X,G,listoffactors,setting)
#pdb.set_trace() 

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
        return np.sum(np.array(Errorlist))
    
#pool=Pool(5)     
#X=tl.tensor(np.random.normal(loc=0,scale=1,size=(20,30,40)))
#G=tl.tensor(np.random.normal(loc=0,scale=1,size=(15,25,35)))
#listoffactors=[tl.tensor(np.random.normal(loc=0,scale=1,size=(20,15))),tl.tensor(np.random.normal(loc=0,scale=1,size=(30,25))),tl.tensor(np.random.normal(loc=0,scale=1,size=(40,35)))]
#setting="Single"
#Err1=Error(X,G,listoffactors,setting,pool)
#
#X=[tl.tensor(np.random.normal(loc=0,scale=1,size=(20,30,40))),tl.tensor(np.random.normal(loc=0,scale=1,size=(20,30,40))),tl.tensor(np.random.normal(loc=0,scale=1,size=(20,30,40)))]
#G=[tl.tensor(np.random.normal(loc=0,scale=1,size=(15,25,35))),tl.tensor(np.random.normal(loc=0,scale=1,size=(15,25,35))),tl.tensor(np.random.normal(loc=0,scale=1,size=(15,25,35)))]
#listoffactors=[tl.tensor(np.random.normal(loc=0,scale=1,size=(20,15))),tl.tensor(np.random.normal(loc=0,scale=1,size=(30,25))),tl.tensor(np.random.normal(loc=0,scale=1,size=(40,35)))]
#setting="MiniBatch"
#Err2=Error(X,G,listoffactors,setting,pool)
#print("Checking")
#print(tl.norm(tl.tensor(np.random.rand(10,20)),2))
#pdb.set_trace() 

def Lineserchstep(X,G,listoffactors,Nonnegative,setting,A,n,Grad,a,b,alpha,theta,pool):
    #This function computes the step with line search
    #All the parameters are of tensor type
    Matrix_list=list(listoffactors)
    r=(np.sqrt(5)-1)/2
    As=a
    Bs=b
    L=Bs-As
    lambda1=As+r*r*L
    lambda2=As+r*L
    nb_iter=0
    F1=1
    F2=0
    Lambda=0
    while(nb_iter<5):
      nb_iter=nb_iter+1
      if(Nonnegative==False):
          
         Matrix_list[n]=A-lambda1*Grad
         #F1=np.power(T.norm(X-Tensor_matrixproduct(G,Matrix_list),2),2)+alpha*(1-theta)*np.power(T.norm(A,2),2)
         #print(type(A))
         #print(A)
         #Secours=(alpha*(1-theta)/2)*np.power(tl.norm(listoffactors[n],2),2)
         #pdb.set_trace()
         F1=Error(X,G,Matrix_list,setting,pool)+(alpha*(1-theta)/2)*np.power(tl.norm(A,2),2)
         Matrix_list[n]=tl.tensor(A-lambda2*Grad)
         F2=Error(X,G,Matrix_list,setting,pool)+(alpha*(1-theta)/2)*np.power(tl.norm(A,2),2)
         
        
      if(Nonnegative==True):
         
         Matrix_list[n]=tl.backend.maximum(A-lambda1*Grad,0)
         #F1=np.power(T.norm(X-Tensor_matrixproduct(G,Matrix_list),2),2)+alpha*(1-theta)*np.power(T.norm(A,2),2)
         F1=Error(X,G,Matrix_list,setting,pool)+(alpha*(1-theta)/2)*np.power(tl.norm(A,2),2)
         
         Matrix_list[n]=tl.backend.maximum(A-lambda2*Grad,0)
         #F2=np.power(T.norm(X-Tensor_matrixproduct(G,Matrix_list),2),2)+alpha*(1-theta)*np.power(T.norm(A,2),2)
         F2=Error(X,G,Matrix_list,setting,pool)+(alpha*(1-theta)/2)*np.power(tl.norm(A,2),2)
            
      if(F1>F2):
          As=lambda1
          lambda1=lambda2
          L=Bs-As
          lambda2=As+r*L
      else:
          Bs=lambda2
          lambda2=lambda1
          L=Bs-As
          lambda1=As+r*r*L
          
      if((L<0.001) or nb_iter>=5):
          Lambda=(Bs+As)/2
    return Lambda


    
def Operations_listmatrices(listofmatrices,operationnature):
    #This function takes a list of matrices and performs some classical operations on its elements.
    #The variable operationnature specifies the operation performed
    #The matrices are of tensor type
    Res=[]
    if (operationnature=="Transpose"):
        for matrix in listofmatrices:
           #element=np.copy(mxnet_backend.to_numpy(matrix))
           element=np.copy(tl.backend.to_numpy(matrix))
           Res.append(tl.tensor(element.T))#computes A.T
        return Res
    
    if(operationnature=="Transposetimes"):
       for matrix in listofmatrices:
           #element=np.copy(mxnet_backend.to_numpy(matrix))
           element=np.copy(tl.backend.to_numpy(matrix))
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
    
#listofmatrices=[T.tensor(np.ones((10,50))),T.tensor(np.ones((10,5)))]    
#operationnature="Transposetimes"
#Res=Operations_listmatrices(listofmatrices,operationnature)
#print(Res[0].shape)
#print(Res[1].shape)
#pdb.set_trace()
           
def TestPositivity_single(X):
    #This function is used to test if the data is inherently positive for a single tensor
    #The parameter is a tensor
    #Size=np.size(np.array(mxnet_backend.to_numpy(X)))
    Size=np.size(tl.backend.to_numpy(X))
    #Bool=np.sum(np.array(mxnet_backend.to_numpy(X)>=0))/Size
    Bool=np.sum(np.array(tl.backend.to_numpy(X)>=0))/Size
    if(Bool!=1):#There is at least one negative elements
      return False
    if(Bool==1):
      return True
#X=tl.tensor(np.maximum(np.random.normal(loc=0,scale=1,size=(20,25,30)),0))
#Res=TestPositivity_single(X)
#pdb.set_trace()

def TestPositivity(X):
    #This function is used to test if all the tensors for a sequence are inherently positive
    Res=True
    for tensors in X:
        res=TestPositivity_single(tensors)
        if(res==False):
            Res=False
            break
    return Res

#Tensor1=tl.tensor(np.random.rand(20,25,30))
#Tensor2=tl.tensor(np.random.rand(20,25,30))
#Tensor3=tl.tensor((-1)*np.random.rand(20,25,30))
#X=[Tensor1,Tensor2,Tensor3]
#Res1=TestPositivity(X)
#print(Res1)
#pdb.set_trace()
      
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
  
#X=tl.tensor(np.random.rand(10,20,30,40)) 
#G=tl.tensor(np.random.rand(5,15,20,25))
#listofmatrices=[tl.tensor(np.ones((10,5))),tl.tensor(np.ones((20,15))),tl.tensor(np.ones((30,20))),tl.tensor(np.ones((40,25)))]
#B=derivativeCore(X,G,listofmatrices)
#print(B.shape)
#pdb.set_trace()
    
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
             G_new=tl.backend.maximum(G_old-step*derivativeCore(X,G_old,listoffactors)+tl.tensor(alpha*theta*np.ones(G_old.shape)),0)
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
       G=pool.map(Sparse_codingSingle,[[X,G,listoffactors,Nonnegative,step,max_iter,alpha,theta,epsilon,l] for l in range (L)])
       G_result=[]
  
       for Goutput in G:
          G_result.append(Goutput)
       return G_result
    
#X=tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40)))
#G_init=tl.tensor(np.random.normal(loc=0,scale=1,size=(5,15,20,25)))
#listoffactors=[tl.tensor(np.random.normal(loc=0,scale=1,size=(10,5))),tl.tensor(np.random.normal(loc=0,scale=1,size=(20,15))),tl.tensor(np.random.normal(loc=0,scale=1,size=(30,20))),tl.tensor(np.random.normal(loc=0,scale=1,size=(40,25)))]
#step=np.power(10,-20,dtype=float)
#max_iter=20
#alpha=np.power(10,3,dtype=float)
#theta=np.power(10,-5,dtype=float)
#epsilon=np.power(10,-10,dtype=float)
#Setting="Single"
#pool=Pool(5)
#Nonnegative=False
#Gresult=Sparse_coding(X,G_init,listoffactors,Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool)
#pdb.set_trace()

#X=[tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40))),tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40))),tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40)))]
#G_init=[tl.tensor(np.random.normal(loc=0,scale=1,size=(5,15,20,25))),tl.tensor(np.random.normal(loc=0,scale=1,size=(5,15,20,25))),tl.tensor(np.random.normal(loc=0,scale=1,size=(5,15,20,25)))]
#listoffactors=[tl.tensor(np.random.normal(loc=0,scale=1,size=(10,5))),tl.tensor(np.random.normal(loc=0,scale=1,size=(20,15))),tl.tensor(np.random.normal(loc=0,scale=1,size=(30,20))),tl.tensor(np.random.normal(loc=0,scale=1,size=(40,25)))]
#step=np.power(10,-20,dtype=float)
#max_iter=20
#alpha=np.power(10,3,dtype=float)
#theta=np.power(10,-5,dtype=float)
#epsilon=np.power(10,-10,dtype=float)
#Setting="MiniBatch"
#pool=Pool(5)
#Nonnegative=False
#Gresult=Sparse_coding(X,G_init,listoffactors,Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool)
#pdb.set_trace()

def Normsingle(args):#The parameter is a tensor
    return tl.norm(args[0][1],2)

def NormSet(X_set,pool):
    L=len(X_set)
    result=pool.map(Normsingle,[[X_set,l] for l in range (L)])
    return np.sum(np.array(result))
#X_set=[tl.tensor(np.random.rand(20,25,30)),tl.tensor(2*np.random.rand(20,25,30)),tl.tensor(3*np.random.rand(20,25,30))]
#pool=Pool(10)
#Result=NormSet(X_set,pool)
#pdb.set_trace()   

def Compute_test_error(X_test,G_init,listoffactors,Nonnegative,step,max_iter,alpha,theta,epsilon,pool):#
    G_test=Sparse_coding(X_test,G_init,listoffactors,Nonnegative,"MiniBatch",step,max_iter,alpha,theta,epsilon,pool)
    Test_error=Error(X_test,G_test,listoffactors,"MiniBatch",pool)
    return Test_error
  

def derivativeDict(X,G,A,listofmatrices,Pold,Qold,setting,alpha,theta,n,t):
    #The function is used to compute the derivative of the objective function with respect the nth dictionary matrix
    #The parameters are tensors
    listoffactors=list(listofmatrices)
    #Pnew=T.tensor(np.copy(mxnet_backend.to_numpy(Pold)))
    #Qnew=T.tensor(np.copy(mxnet_backend.to_numpy(Qold)))
    Pnew=tl.tensor(Pold)
    Qnew=tl.tensor(Qold)

    if(setting=="Single"):        
       listoffactors[n]=np.identity(X.shape[n])   
       WidehatX=Tensor_matrixproduct(X,Operations_listmatrices(listoffactors,"Transpose"))       
       listoffactors[n]=tl.tensor(np.identity(G.shape[n]))
       B=Tensor_matrixproduct(G,listoffactors) 
       #B=unfold(Offset(G),mode)
       #pdb.set_trace()
       
       Pnew=Pnew+tl.tensor(np.dot(mxnet_backend.to_numpy(unfold(WidehatX,n)),mxnet_backend.to_numpy(unfold(G,n)).T))
       Qnew=Qnew+tl.tensor(np.dot(mxnet_backend.to_numpy(unfold(B,n)),mxnet_backend.to_numpy(unfold(B,n)).T))
      
       Res=-Pnew/t+tl.tensor(np.dot(mxnet_backend.to_numpy(A),mxnet_backend.to_numpy(Qnew))/t)+alpha*(1-theta)*A
      
       return Res
   
    if(setting=="MiniBatch"):
      
       rho=len(X)
       for r in range(rho):
          listoffactors[n]=tl.tensor(np.identity(X[r].shape[n]))
          
          WidehatX=Tensor_matrixproduct(X[r],Operations_listmatrices(listoffactors,"Transpose")) 
          
          Pnew=Pnew+tl.tensor(np.dot(mxnet_backend.to_numpy(unfold(WidehatX,n)),mxnet_backend.to_numpy(unfold(G[r],n)).T)) 
          
          listoffactors[n]=tl.tensor(np.identity(G[r].shape[n]))
       
          B=Tensor_matrixproduct(G[r],listoffactors) 
  
          Qnew=Qnew+tl.tensor(np.dot(mxnet_backend.to_numpy(unfold(B,n)),mxnet_backend.to_numpy(unfold(B,n).T)))
    
       Res=-Pnew/t+tl.tensor(np.dot(mxnet_backend.to_numpy(A),mxnet_backend.to_numpy(Qnew))/t)+alpha*(1-theta)*A
       return Res

#X=tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40)))
#G=tl.tensor(np.random.normal(loc=0,scale=1,size=(10,15,20,25)))
#n=3
#A=tl.tensor(np.ones((40,25)))
#setting="Single"
#listofmatrices=[tl.tensor(np.random.normal(loc=0,scale=1,size=(10,10))),tl.tensor(np.random.normal(loc=0,scale=1,size=(20,15))),tl.tensor(np.random.normal(loc=0,scale=1,size=(30,20))),tl.tensor(np.random.normal(loc=0,scale=1,size=(40,25)))]
#Pold=tl.tensor(np.random.rand(40,25))
#
#Qold=tl.tensor(np.random.rand(25,25))
#theta=0.1
#alpha=0.1
#t=2
#Outcome=derivativeDict(X,G,A,listofmatrices,Pold,Qold,setting,alpha,theta,n,t)
#pdb.set_trace()

#X=[tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40))),tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40))),tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40)))]
#G=[tl.tensor(np.random.normal(loc=0,scale=1,size=(10,15,20,25))),tl.tensor(np.random.normal(loc=0,scale=1,size=(10,15,20,25))),tl.tensor(np.random.normal(loc=0,scale=1,size=(10,15,20,25)))]
#n=3
#A=tl.tensor(np.ones((40,25)))
#setting="MiniBatch"
#listofmatrices=[tl.tensor(np.random.normal(loc=0,scale=1,size=(10,10))),tl.tensor(np.random.normal(loc=0,scale=1,size=(20,15))),tl.tensor(np.random.normal(loc=0,scale=1,size=(30,20))),tl.tensor(np.random.normal(loc=0,scale=1,size=(40,25)))]
#Pold=tl.tensor(np.random.rand(40,25))
#Qold=tl.tensor(np.random.rand(25,25))
#theta=0.1
#alpha=0.1
#t=2
#Outcome=derivativeDict(X,G,A,listofmatrices,Pold,Qold,setting,alpha,theta,n,t)
#pdb.set_trace()

def NormSingleTensor(args):
    return np.power(tl.norm(args[0][1],2),2)

def NormSumTensors(X_set,pool):
    L=len(X_set)
    print(L)
    Error=pool.map(NormSingleTensor,[[X_set,l] for l in range(L)])
    error=np.sum(np.array(Error))
    return error

def Norm(X,setting,pool):
    if(setting=="Single"):
        return np.power(tl.norm(X,2),2)
    if(setting=="MiniBatch"):
        return NormSumTensors(X,pool)
    
def Dictionary_update(X,G,listofmatrices,Nonnegative,Pold,Qold,setting,alpha,theta,step,max_iter,epsilon,n,t,period,pool):
    #This function is used to perform the gradient descent with line search while enforcing nonnegativity if this option is chosen
    #All the parameters are tensors
    Anew=tl.tensor(listofmatrices[n])
    Aold=tl.tensor(np.zeros(Anew.shape))
    Aresult=tl.tensor(np.zeros(Anew.shape))
    listoffactors=list(listofmatrices) 
    error=Error(X,G,listoffactors,setting,pool)#+(alpha*(1-theta)/2)*np.power(T.norm(Anew,2),2)
    
    error_list=[error]
    #previous_error=0
    nb_iter=0
    while(nb_iter<=max_iter):
         nb_iter=nb_iter+1
         #previous_error=error
         Aold=Anew
         if(Nonnegative==True):
             derivative=derivativeDict(X,G,Aold,listofmatrices,Pold,Qold,setting,alpha,theta,n,t)
             if( (nb_iter%period)==0 or (nb_iter==1) ):
                 a=step/10
                 b=step
                 step=Lineserchstep(X,G,listoffactors,Nonnegative,setting,Aold,n,derivative,a,b,alpha,theta,pool)
                      
             #Anew=np.maximum(Aold-step*derivativeDict(X,G,Aold,listofmatrices,Pold,Qold,setting,alpha,theta,n,t),0)
 
             Anew=tl.backend.maximum(Aold-step*derivative,0)
             Anew=Anew/np.maximum(tl.norm(Anew,2),epsilon)
             
         if(Nonnegative==False):
             derivative=derivativeDict(X,G,Aold,listofmatrices,Pold,Qold,setting,alpha,theta,n,t)
             if( (nb_iter%period)==0 or (nb_iter==1) ):
                 a=step/10
                 b=step
                 step=Lineserchstep(X,G,listoffactors,Nonnegative,setting,Aold,n,derivative,a,b,alpha,theta,pool)
                      #Lineserchstep(X,G,listoffactors,Nonnegative,setting,A,n,Grad,a,b,alpha,theta,pool)
             Anew=Aold-step*derivativeDict(X,G,Aold,listofmatrices,Pold,Qold,setting,alpha,theta,n,t)
             Anew=Anew/np.maximum(tl.norm(Anew,2),epsilon)
             
         #listoffactors[n]=Anew
         #error=T.norm(X-Tensor_matrixproduct(G,listoffactors),2)+alpha*(1-theta)*T.norm(Anew,2)
         #pdb.set_trace()
         error=Error(X,G,listoffactors,setting,pool)#+alpha*(1-theta)*T.norm(Anew,2)
         error_list.append(error)
         Aresult=Anew
         #print(Norm(X,setting,pool))
         #pdb.set_trace()
         #if(np.abs(previous_error-error)/error<epsilon):
         if(error/Norm(X,setting,pool)<epsilon):
             Aresult=Aold
             error_list=error_list[0:len(error_list)-1]
             break         
    return Aresult,error_list,nb_iter 
      

#X=tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40)))
#G=tl.tensor(np.random.normal(loc=0,scale=1,size=(10,15,20,25)))
#n=1
#pool=Pool(10)
#Pold=tl.tensor(np.random.rand(20,15))
#Qold=tl.tensor(np.random.rand(15,15))
#setting="Single"
#Nonnegative=False
#max_iter=20
#epsilon=0.01
#period=5
#t=5
#alpha=0.01
#theta=0.1
#step=np.power(10,-10,dtype=float)
#listofmatrices=[tl.tensor(np.random.normal(loc=0,scale=1,size=(10,10))),tl.tensor(np.random.normal(loc=0,scale=1,size=(20,15))),tl.tensor(np.random.normal(loc=0,scale=1,size=(30,20))),tl.tensor(np.random.normal(loc=0,scale=1,size=(40,25)))]
#A,error_list,nb_iter=Dictionary_update(X,G,listofmatrices,Nonnegative,Pold,Qold,setting,alpha,theta,step,max_iter,epsilon,n,t,period,pool)
#pdb.set_trace()
    
#X=[tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40))),tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40))),tl.tensor(np.random.normal(loc=0,scale=1/5,size=(10,20,30,40)))]
#G=[tl.tensor(np.random.normal(loc=0,scale=1,size=(10,15,20,25))),tl.tensor(np.random.normal(loc=0,scale=1,size=(10,15,20,25))),tl.tensor(np.random.normal(loc=0,scale=1,size=(10,15,20,25)))]
#n=1
#Pold=tl.tensor(np.random.rand(20,15))
#Qold=tl.tensor(np.random.rand(15,15))
#setting="MiniBatch"
#Nonnegative=False
#pool=Pool(10)
#period=5
#max_iter=20
#epsilon=0.01
#t=5
#alpha=0.01
#theta=0.1
#step=np.power(10,-10,dtype=float)
#listofmatrices=[tl.tensor(np.random.normal(loc=0,scale=1,size=(10,10))),tl.tensor(np.random.normal(loc=0,scale=1,size=(20,15))),tl.tensor(np.random.normal(loc=0,scale=1,size=(30,20))),tl.tensor(np.random.normal(loc=0,scale=1,size=(40,25)))]
#A,error_list,nb_iter=Dictionary_update(X,G,listofmatrices,Nonnegative,Pold,Qold,setting,alpha,theta,step,max_iter,epsilon,n,t,period,pool)
#pdb.set_trace()


def NORM(Tensor,Setting,p):
    #The function computes the sum of the square of the norm for a sequence of matrices
    #The paramters are tensors
    if(Setting=="Single"):
        return tl.norm(Tensor,p)
    if(Setting=="MiniBatch"):
        if(p==1):
           return np.sum(np.array(Operations_listmatrices(Tensor,"NormI")))
        if(p==2):
           return np.sum(np.array(Operations_listmatrices(Tensor,"NormII")))
       
def ChooseoperationBCD(X,Setting):
    #This function is used to turn the arrays in a list into tensor type elements
    #The elements are arrays
    if(Setting=="Single"):
        return tl.tensor(X)
    if(Setting=="MiniBatch"):
        return Operations_listmatrices(X,"Tensorize")

def CyclicBlocCoordinateTucker_single(X,Coretensorsize,Pre_existingfactors,Pre_existingG,Pre_existingP,Pre_existingQ,Nonnegative,Setting,t,step,alpha,theta,max_iter,epsilon,period,pool):#All the parameters are arrays,
    #This function is used to update the factors when the newly incoming tensor is a single tensor
    #All the parameters are arrays
    if(Setting=="Single"):
       N=len(list(X.shape))
    
       Inferredfactorsnew=list(Pre_existingfactors)
    
       listofmatrices=list(Pre_existingfactors)
    
       Inferredfactorsresult=[]
    
       error=Error(tl.tensor(X),tl.tensor(Pre_existingG),Operations_listmatrices(Inferredfactorsnew,"Tensorize"),Setting,pool)#+alpha*theta*NORM(Pre_existingG,Setting,1)+(alpha*(1-theta)/2)*np.sum(np.array(Operations_listmatrices(Inferredfactorsnew,"NormII")))
       
    
       Inferredfactorsold=[]
       nb_iter=0    
       
       listoffactorsG=list(Pre_existingfactors)
          
       G=Sparse_coding(tl.tensor(X),tl.tensor(Pre_existingG),Operations_listmatrices(listoffactorsG,"Tensorize"),Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool) 
          
       while(nb_iter<max_iter):
        
          nb_iter=nb_iter+1
       
          
          Inferredfactorsold=Inferredfactorsnew
       
          #We update the dictionary matrices for an instance tensor
          for n in range(N):
          
             Pold=Pre_existingP[n]
          
             Qold=Pre_existingQ[n]
          
             An,error_list,nb_iter=Dictionary_update(ChooseoperationBCD(X,Setting),ChooseoperationBCD(G,Setting),Operations_listmatrices(listofmatrices,"Tensorize"),Nonnegative,tl.tensor(Pold),tl.tensor(Qold),Setting,alpha,theta,step,max_iter,epsilon,n,t,period,pool)
          
             listofmatrices[n]=An
          
             Inferredfactorsnew[n]=An
        
          error=Error(tl.tensor(X),tl.tensor(Pre_existingG),Operations_listmatrices(Inferredfactorsnew,"Tensorize"),Setting,pool)#+alpha*theta*NORM(Pre_existingG,Setting,1)+(alpha*(1-theta)/2)*np.sum(np.array(Operations_listmatrices(Inferredfactorsnew,"NormII")))
              
          Inferredfactorsresult=Inferredfactorsnew
          #if(np.abs(previous_error-error)/error<epsilon):
          if(np.sqrt(error)/tl.norm(tl.tensor(X),2)<epsilon):
             Inferredfactorsresult=Inferredfactorsold
             break
   
       G_init=Pre_existingG
       Gresult=Sparse_coding(tl.tensor(X),tl.tensor(G_init),Inferredfactorsresult,Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool)
       return Gresult,Inferredfactorsresult
   
    if(Setting=="MiniBatch"):
       
       N=len(list(X[0].shape))
    
       #Inferredfactorsnew=list(Pre_existingfactors)
    
       #listofmatrices=list(Pre_existingfactors)
       
       Inferredfactorsnew=list(Pre_existingfactors)
       
       listofmatrices=Pre_existingfactors
    
       Inferredfactorsresult=[]
       print("Point I in MiniBatch")
       error=Error(Operations_listmatrices(X,"Tensorize"),Operations_listmatrices(Pre_existingG,"Tensorize"),Operations_listmatrices(Inferredfactorsnew,"Tensorize"),Setting,pool)#+alpha*theta*NORM(Pre_existingG,Setting,1)+(alpha*(1-theta)/2)*np.sum(np.array(Operations_listmatrices(Inferredfactorsnew,"NormII")))
       print("Point II in MiniBatch")
       Inferredfactorsold=[]
       
       listoffactorsG=list(Pre_existingfactors)
          
       #G=Sparse_coding(Operations_listmatrices(X,"Tensorize"),Operations_listmatrices(Pre_existingG,"Tensorize"),Operations_listmatrices(listoffactorsG,"Tensorize"),Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool) 
           
       G=Sparse_coding(X,Pre_existingG,listoffactorsG,Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool) 
           
       nb_iter=0    
       while(nb_iter<max_iter):
        
          nb_iter=nb_iter+1
       
          Inferredfactorsold=Inferredfactorsnew
       
          #previous_error=error
          #We update the dictionary matrices for an instance tensor
          for n in range(N):
          
             Pold=Pre_existingP[n]
          
             Qold=Pre_existingQ[n]
               
             print("Point III in MiniBatch")
             
             An,error_list,nb_iter=Dictionary_update(ChooseoperationBCD(X,Setting),ChooseoperationBCD(G,Setting),Operations_listmatrices(listofmatrices,"Tensorize"),Nonnegative,tl.tensor(Pold),tl.tensor(Qold),Setting,alpha,theta,step,max_iter,epsilon,n,t,period,pool)
          
             listofmatrices[n]=An
          
             Inferredfactorsnew[n]=An
        
          error=Error(Operations_listmatrices(X,"Tensorize"),Operations_listmatrices(Pre_existingG,"Tensorize"),Operations_listmatrices(Inferredfactorsnew,"Tensorize"),Setting,pool)#+alpha*theta*NORM(Pre_existingG,Setting,1)+(alpha*(1-theta)/2)*np.sum(np.array(Operations_listmatrices(Inferredfactorsnew,"NormII")))
          print("Point IV in MiniBatch")   
          Inferredfactorsresult=Inferredfactorsnew
          #if(np.abs(previous_error-error)/error<epsilon):
          if(np.sqrt(error)/NORM(X,Setting,2)<epsilon):
             Inferredfactorsresult=Inferredfactorsold
             break
   
       print("Point V in MiniBatch")
       print("The number of iterations is")
       print(nb_iter)
       Gresult=Sparse_coding(Operations_listmatrices(X,"Tensorize"),Operations_listmatrices(Pre_existingG,"Tensorize"),Operations_listmatrices(Inferredfactorsresult,"Tensorize"),Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool)
       print("Point VI in MiniBatch")
       return Gresult,Inferredfactorsresult 

#X=np.random.normal(loc=0,scale=1,size=(30,40,50))
#Coretensorsize=np.array([15,20,25],dtype=int)
#Pre_existingfactors=[np.random.rand(30,15),np.random.rand(40,20),np.random.rand(50,25)]
#Pre_existingG=np.random.normal(loc=0,scale=1,size=(15,20,25))
#Pre_existingP=[np.random.rand(30,15),np.random.rand(40,20),np.random.rand(50,25)]
#Pre_existingQ=[np.random.rand(15,15),np.random.rand(20,20),np.random.rand(25,25)]
#LastObjectiveerror=0.01
#pool=Pool(10)
#Setting="Single"
#t=10
#step=np.power(10,-10,dtype=float)
#alpha=0.01
#theta=0.1
#period=40
#Nonnegative=False
#max_iter=200
#epsilon=np.power(10,-10,dtype=float)
#Gresult,Inferredfactorsresult=CyclicBlocCoordinateTucker_single(X,Coretensorsize,Pre_existingfactors,Pre_existingG,Pre_existingP,Pre_existingQ,Nonnegative,Setting,t,step,alpha,theta,max_iter,epsilon,period,pool)
#pdb.set_trace()



##MiniBatch
#X=[np.random.normal(loc=0,scale=1,size=(30,40,50)),np.random.normal(loc=0,scale=1,size=(30,40,50)),np.random.normal(loc=0,scale=1,size=(30,40,50))]
#Coretensorsize=np.array([15,20,25],dtype=int)
#Pre_existingfactors=[np.random.rand(30,15),np.random.rand(40,20),np.random.rand(50,25)]
#Pre_existingG=[np.random.normal(loc=0,scale=1,size=(15,20,25)),np.random.normal(loc=0,scale=1,size=(15,20,25)),np.random.normal(loc=0,scale=1,size=(15,20,25))]
#Pre_existingP=[np.random.rand(30,15),np.random.rand(40,20),np.random.rand(50,25)]
#Pre_existingQ=[np.random.rand(15,15),np.random.rand(20,20),np.random.rand(25,25)]
#LastObjectiveerror=0.01
#Setting="MiniBatch"
#t=100
#pool=Pool(10)
#period=40
#Nonnegative=False
#step=np.power(10,-10,dtype=float)
#alpha=0.01
#theta=0.1
#max_iter=200
#epsilon=np.power(10,-10,dtype=float)
#Gresult,Inferredfactorsresult=CyclicBlocCoordinateTucker_single(X,Coretensorsize,Pre_existingfactors,Pre_existingG,Pre_existingP,Pre_existingQ,Nonnegative,Setting,t,step,alpha,theta,max_iter,epsilon,period,pool)
#pdb.set_trace()



def SplitToMinibatch(Tensorset,sizeminibatch):
    #This function is used to split a set of tensors 
    #The elements are tensors
    K=len(sizeminibatch)#The number of minibatches
    Card=len(Tensorset)
    if(Card<K):
        raise Exception("The number of minibatches cannot be greater than the number of instances")
    if(Card!=np.sum(np.array(sizeminibatch))):
        raise Exception("There is inconsistency between the number of instances and the number indicated by the minibatches sizes")
    Res=[]
    pos1=0
    pos2=0
    for k in range(K):
        pos2=pos2+sizeminibatch[k]
        Res.append(Tensorset[pos1:pos2])
        pos1=pos2
    return Res

#Tensorset=[]
#for n in range(10):
#    Tensorset.append(tl.tensor(n*np.random.normal(loc=0,scale=1,size=(20,30,40))))
#sizeminibatch=[2,5,3]
#Res=SplitToMinibatch(Tensorset,sizeminibatch)
#print(len(Res[0]))
#print(len(Res[1]))
#print(len(Res[2]))
#pdb.set_trace()

def CyclicBlocCoordinateTucker_set(X_set,Coretensorsize,Pre_existingfactors,Pre_existingG_set,Pre_existingP,Pre_existingQ,Nonnegative,Reprojectornot,Setting,Minibatchnumber,step,alpha,theta,max_iter,epsilon,period,pool):#All the parameters are arrays,
    #This function is used to perform the online setting either for single or minibatch processing
    if(Nonnegative==True):
        Positivity=TestPositivity(X_set)
        if(Positivity==False):
            raise Exception("You decide to perform a nonnegative decomposition while some of your tensors present negative entries")
    
    L=len(X_set)

    if(Setting=="Single"):
        for t in range(L):
          
           X=X_set[t]
           Pre_existingG=Pre_existingG_set[t]
          
           Gresult,Inferredfactorsresult=CyclicBlocCoordinateTucker_single(X,Coretensorsize,Pre_existingfactors,Pre_existingG,Pre_existingP,Pre_existingQ,Nonnegative,Setting,t+1,step,alpha,theta,max_iter,epsilon,period,pool)
           #print(len(Inferredfactorsresult))
           
           N=len(list(X.shape))  
           Pre_existingG=np.copy(Gresult)
           G=tl.tensor(Gresult)
           for n in range(N):
             listoffactors=list(Inferredfactorsresult)
             #print("Point I")
             #print(listoffactors[n].shape)
             listoffactors[n]=np.identity(X.shape[n]) 
             #print("Point II")
             #print(listoffactors[n].shape)
             WidehatX=Tensor_matrixproduct(X,Operations_listmatrices(listoffactors,"Transpose"))       
             listoffactors[n]=tl.tensor(np.identity(G.shape[n]))
             B=Tensor_matrixproduct(tl.tensor(G),listoffactors) 
             
             Pre_existingP[n]=Pre_existingP[n]+tl.backend.dot(unfold(WidehatX,n),unfold(G,n).T)
             
             Pre_existingQ[n]=Pre_existingQ[n]+tl.backend.dot(unfold(B,n),unfold(B,n).T)             
        
        if(Reprojectornot==True):
            Gresult=[]
            for t in range(L):
              X=X_set[t]
              G_init=Pre_existingG_set[t]
              G=Sparse_coding(tl.tensor(X),tl.tensor(G_init),Operations_listmatrices(Inferredfactorsresult,"Tensorize"),Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool)
              Gresult.append(G)
                   
            return Gresult,Inferredfactorsresult
        if(Reprojectornot==False):
            return Inferredfactorsresult
    
    
    if(Setting=="MiniBatch"):
        X_setdivided=SplitToMinibatch(X_set,Minibatchnumber)
        Pre_existingGsetdivided=SplitToMinibatch(Pre_existingG_set,Minibatchnumber)
        Pre_existingPold=[]
        Pre_existingPnew=list(Pre_existingP)
        Pre_existingQold=[]
        Pre_existingQnew=list(Pre_existingQ)
        #for mininb in range(Minibatchnumber):
        for mininb in range(len(Minibatchnumber)):
            print("We are minibatch")
            print("The minibatch processed is")
            print(mininb)
            X_minibatch=X_setdivided[mininb]
            Pre_existingG_minibatch=Pre_existingGsetdivided[mininb]
            Pre_existingPold=Pre_existingPnew
            Pre_existingQold=Pre_existingQnew            
         
            Gresult,Inferredfactorsresult=CyclicBlocCoordinateTucker_single(X_minibatch,Coretensorsize,Pre_existingfactors,Pre_existingG_minibatch,Pre_existingPold,Pre_existingQold,Nonnegative,Setting,mininb+1,step,alpha,theta,max_iter,epsilon,period,pool)
            
            if(mininb!=len(Minibatchnumber)-1):
               X_minibatchold=Operations_listmatrices(X_setdivided[mininb],"Tensorize")
               N=len(list(X_minibatchold[0].shape))
               Inferredfactorsresult=Operations_listmatrices(Inferredfactorsresult,"Tensorize")
               minibatchsize=len(X_minibatchold)
               N=len(X_minibatchold[0].shape)
             
               Gactivationcoeff=list(Gresult)
               
               for n in range(N): 
                   for r in range(minibatchsize):                              
                     X=X_minibatchold[r]
                     G=Gactivationcoeff[r]
                     listoffactors=list(Inferredfactorsresult)
                     
                     listoffactors[n]=np.identity(X.shape[n]) 
                      
                     WidehatX=Tensor_matrixproduct(tl.tensor(X),Operations_listmatrices(listoffactors,"Transpose"))       
                     listoffactors[n]=np.identity(G.shape[n])
             
                     B=Tensor_matrixproduct(tl.tensor(G),Operations_listmatrices(listoffactors,"Tensorize"))        
                     Pre_existingPnew[n]=Pre_existingPold[n]+tl.backend.dot(unfold(WidehatX,n),unfold(G,n).T)       
                     Pre_existingQnew[n]=Pre_existingQold[n]+tl.backend.dot(unfold(B,n),unfold(B,n).T) 
                   

        if(Reprojectornot==True):
            
            for mininb in range(len(Minibatchnumber)):
               X_minibatch=Operations_listmatrices(X_setdivided[mininb],"Tensorize")
               G_init=Operations_listmatrices(Pre_existingGsetdivided[mininb],"Tensorize")
               
               G=Sparse_coding(X_minibatch,G_init,Inferredfactorsresult,Nonnegative,Setting,step,max_iter,alpha,theta,epsilon,pool)
           
               for activation_coeff in G:
              
                  Gresult.append(activation_coeff)
             
            return Gresult,Inferredfactorsresult
        if(Reprojectornot==False):
            return Inferredfactorsresult
        

     
def CyclicBlocCoordinateTucker_setWithPredefinedEpochs(X_set,Coretensorsize,Pre_existingfactors,Pre_existingG_set,Pre_existingP,Pre_existingQ,Nonnegative,Reprojectornot,Setting,Minibatchnumber,step,alpha,theta,max_iter,epsilon,period,nbepochs,pool):#All the parameters are arrays,
    
    Listoffactorsnew=list(Pre_existingfactors)
    Listoffactorsold=[]
    G_setnew=list(Pre_existingG_set)
    G_setold=[]

    for nepoch in range(nbepochs):
        Listoffactorsold= Listoffactorsnew
        G_setold=G_setnew
        np.random.seed(nepoch)
        X_setreshuffled=TensorDataDrawnRandomly(X_set)
        
        if(Reprojectornot==True):
           G_setnew,Listoffactorsnew=CyclicBlocCoordinateTucker_set(X_setreshuffled,Coretensorsize,Listoffactorsold,G_setold,Pre_existingP,Pre_existingQ,Nonnegative,Reprojectornot,Setting,Minibatchnumber,step,alpha,theta,max_iter,epsilon,period,pool)
           return G_setnew,Listoffactorsnew
        if(Reprojectornot==False):
           Listoffactorsnew=CyclicBlocCoordinateTucker_set(X_setreshuffled,Coretensorsize,Listoffactorsold,G_setold,Pre_existingP,Pre_existingQ,Nonnegative,Reprojectornot,Setting,Minibatchnumber,step,alpha,theta,max_iter,epsilon,period,pool)
           return Listoffactorsnew
  

def FittingErrorComputation(X_set,G_set,listoffactors,pool):
    L=len(X_set)
    Fitting_error=[]
    for t in range(L):
        Fitting_error.append(Error(tl.tensor(X_set[t]),tl.tensor(G_set[t]),listoffactors,"Single",pool))
    return Fitting_error
        
       
##TestI: this part is checked
#Nonnegative=False
#pool=Pool(10)
#period=100
#X_set=[np.random.normal(loc=0,scale=1,size=(30,40,50)),2*np.random.normal(loc=0,scale=2,size=(30,40,50)),3*np.random.normal(loc=0,scale=4,size=(30,40,50)),4*np.random.normal(loc=0,scale=8,size=(30,40,50))]
#Coretensorsize=np.array([15,20,25],dtype=int)
#Pre_existingfactors=[np.random.rand(30,15),np.random.rand(40,20),np.random.rand(50,25)]
#Pre_existingG_set=[np.random.normal(loc=0,scale=1,size=(15,20,25)),np.random.normal(loc=0,scale=1,size=(15,20,25)),np.random.normal(loc=0,scale=1,size=(15,20,25)),np.random.normal(loc=0,scale=1,size=(15,20,25))]
#Pre_existingP=[np.random.rand(30,15),np.random.rand(40,20),np.random.rand(50,25)]
#Pre_existingQ=[np.random.rand(15,15),np.random.rand(20,20),np.random.rand(25,25)]
#Setting="Single"
#Reprojectornot=True
#step=np.power(10,-5,dtype=float)
#alpha=0.01