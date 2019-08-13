#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 04:23:44 2018

@author: Traoreabraham
"""


import numpy as np
from multiprocessing import Pool
import scipy
import sys
sys.path.append("..")
import tensorly as tl
tl.set_backend('numpy')
from tensorly.base import unfold


import pdb
from tensorly import tenalg
from tensorly.backend import mxnet_backend
#np.seterr(all='ignore')
np.random.seed(1)
#np.random.seed(100): the results were obtained from this seed
#import random
#sys.path.append("/home/scr/etu/sil821/traorabr/OnlineTensorDictionaryLearning/")
#sys.path.append("/Users/Traoreabraham/Desktop/OnlineTensorDictionaryLearning/")

from MiscellaneousFunctions.OnlineTensor import CyclicBlocCoordinateTucker_setWithPredefinedEpochs
from MiscellaneousFunctions.MethodsTSPen import GenerateTensorsGeneral

def Nonnegativepart(Elements):#The parameters are arrays
    result=[]
    for element in Elements:
        result.append(np.maximum(element,0))
    return result

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
    
def FittingErrorComputation(X_set,G_set,listoffactors):
    L=len(X_set)
    Fitting_error=[]
    for t in range(L):
        Fitting_error.append(Error(tl.tensor(X_set[t]),tl.tensor(G_set[t]),listoffactors,"Single"))
    return Fitting_error


def Proximal_operator(X,step):#The parameter is a tensor
    Res=np.copy(mxnet_backend.to_numpy(X))
    Res=np.sign(Res)*np.maximum(np.abs(Res)-step,0)
    return tl.tensor(Res)


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
       
def derivativeCore(X,G,listofmatrices):#The entries are tensors 
    #Firstterm=T.tensor(np.copy(mxnet_backend.to_numpy(X)))
    Firstterm=tl.tensor(X)
    Firstterm=Tensor_matrixproduct(Firstterm,Operations_listmatrices(listofmatrices,"Transpose"))      
    #Secondterm=T.tensor(np.copy(mxnet_backend.to_numpy(G)))
    Secondterm=tl.tensor(G)
    Secondterm=Tensor_matrixproduct(Secondterm,Operations_listmatrices(listofmatrices,"Transposetimes"))           
    Res=Firstterm-Secondterm
    return Res

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


def Tensor_matrixproduct(X,listoffactors):#The parameters are tensors(tensor and matrices)
    
    Res=tl.tensor(np.copy(mxnet_backend.to_numpy(X)))
    
    mode=-1
    for matrix in listoffactors:
        mode=mode+1
        
        Res=tenalg.mode_dot(Res,matrix,mode) 
       
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

      #previous_error=0
      nb_iter=0
      error_list=[error]    
      while(nb_iter<=max_iter):
          nb_iter=nb_iter+1
          #previous_error=error
          G_old=G_new
          G_new=G_old-step*derivativeCore(X,G_old,listoffactors)
          if(Nonnegative==True):
         
             G_new=tl.tensor(np.maximum(mxnet_backend.to_numpy(G_old-step*(derivativeCore(tl.tensor(X),G_old,listoffactors)))+alpha*theta*np.ones(G_old.shape),0))
        
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
  
def TuckerBatch(X,Coretensorsize,max_iter,listoffactorsinit,Ginit,Nonnegative,Reprojectornot,alpha,theta,step,epsilon):
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

def GenerateTensorsNonnegative(Numberofexamples,randomseed):
    np.random.seed(randomseed)
    Xtrain=np.random.rand(Numberofexamples,30,40,50)
    Coretensorsize=np.array([Numberofexamples,20,20,20])
    Greal=np.maximum(np.random.normal(loc=0,scale=1,size=Coretensorsize),0)
    #The line below changes
    listoffactorsreal=[np.random.normal(loc=0,scale=1/10,size=(30,20)),np.random.normal(loc=0,scale=1/10,size=(40,20)),np.random.normal(loc=0,scale=1/10,size=(50,20))]         
    listoffactorsreal=Nonnegativepart(listoffactorsreal)
    for n in range(Numberofexamples):
       Xtrain[n,:,:,:]=mxnet_backend.to_numpy(Tensor_matrixproduct(tl.tensor(Greal[n,:,:,:]),Operations_listmatrices(listoffactorsreal,"Tensorize")))     
    #Xtrain=Xtrain/T.norm(T.tensor(Xtrain),2)
    return Xtrain


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

def Split_into_two_subsets(X_set,trainratio):
    nbtrainsamples=int(trainratio*len(X_set))
    Xtrainset=X_set[0:nbtrainsamples]
    Xtestset=X_set[nbtrainsamples:len(X_set)]
    return Xtrainset,Xtestset
        
    
    
def ExperimentToyGeneral(etavalues,Nonnegative,Numberofexamples,Minibatchsize,max_iter,step,alpha,theta,nbepochs,randomarray,trainratio,period,pool):    
    L=len(etavalues)
    Nbmean=len(randomarray)   
    RMSEsingle=np.zeros(L)    
    StdsingleRMSE=np.zeros(L)
    Fittingsingle=np.zeros(L)   
    MREsingle=np.zeros(L)    
    
    Stdsinglefitting=np.zeros(L)   
      
    Stdsinglemre=np.zeros(L)   
   
    for l in range(L):
      eta=etavalues[l]    
      Stdsignlermselist=[]    
      Stdsignlefittlist=[]      
      Stdsinglemrelist=[]   
     
      #for m in range(Nbmean):
      for k in range(len(randomarray)):
        
         m=randomarray[k]
         print("The noise number is")
         print(k+1)
         
         X_set=GenerateTensorsGeneral(Numberofexamples,eta,m)

         
         Xtrain_set,Xtest_set=Split_into_two_subsets(X_set,trainratio)
         Xtrain=np.zeros((len(Xtrain_set),30,40,50))
         Xtest=np.zeros((len(Xtest_set),30,40,50))
         for t in range(len(Xtrain_set)):
             Xtrain[t,:,:,:]=Xtrain_set[t]  
         for t in range(len(Xtest_set)):
             Xtest[t,:,:,:]=Xtest_set[t]
                 
         Xtest_set=Operations_listmatrices(Xtest_set,"Tensorize")
       
         epsilon=np.power(10,-15,dtype=float)     
         
         Coretensorsize=np.array([len(Xtrain_set),eta,eta,eta])# The first dimensions must be equal for mathematical coherence purpose
         Ginittrain=np.random.normal(loc=0,scale=1/10,size=([len(Xtrain_set),eta,eta,eta]))    #np.random.normal(loc=0,scale=1/100,size=Coretensorsize)        
        
         Ginittest=np.random.normal(loc=0,scale=1/10,size=(len(Xtest_set),eta,eta,eta))
         Reprojectornot=False
    
         Pre_existingG_settrain=[]
         for n in range(len(Xtrain_set)):
            Pre_existingG_settrain.append(Ginittrain[n,:,:,:])
            
      
         Pre_existingG_settest=[]
         for n in range(len(Xtest_set)):
            Pre_existingG_settest.append(Ginittest[n,:,:,:])
         
         Ltest=len(Xtest_set)
         
         
         Pre_existingfactors=[np.random.normal(loc=0,scale=1/100,size=(30,eta)),np.random.normal(loc=0,scale=1/100,size=(40,eta)),np.random.normal(loc=0,scale=1/100,size=(50,eta))]
         Pre_existingP=[np.random.normal(loc=0,scale=1/100,size=(30,eta)),np.random.normal(loc=0,scale=1/100,size=(40,eta)),np.random.normal(loc=0,scale=1/100,size=(50,eta))]
         Pre_existingQ=[np.random.normal(loc=0,scale=1/100,size=(eta,eta)),np.random.normal(loc=0,scale=1/100,size=(eta,eta)),np.random.normal(loc=0,scale=1/100,size=(eta,eta))]
         

         Coretensorsize=np.array([eta,eta,eta])
         
         Setting="Single"
        
         listoffactorsresult2=CyclicBlocCoordinateTucker_setWithPredefinedEpochs(Xtrain_set,Coretensorsize,Pre_existingfactors,Pre_existingG_settrain,Pre_existingP,Pre_existingQ,Nonnegative,Reprojectornot,Setting,Minibatchsize,step,alpha,theta,max_iter,epsilon,period,nbepochs,pool)
         Xtest_set=Operations_listmatrices(Xtest_set,"Tensorize")
         
         Gtest2=Sparse_coding(Xtest_set,Operations_listmatrices(Pre_existingG_settest,"Tensorize"),listoffactorsresult2,Nonnegative,"MiniBatch",step,max_iter,alpha,theta,epsilon,pool)
         error2=Error(Xtest_set,Gtest2,listoffactorsresult2,"MiniBatch",pool)
         fittingerror2=error2/Ltest
         #pdb.set_trace()
         rmse2=np.sqrt(error2/Ltest)
         mresingle=Mean_relative_error(Xtest_set,Gtest2,Operations_listmatrices(listoffactorsresult2,"Tensorize"),"MiniBatch",pool)
         
         RMSEsingle[l]=RMSEsingle[l]+rmse2
                                
         Fittingsingle[l]=Fittingsingle[l]+fittingerror2
               
         MREsingle[l]=MREsingle[l]+mresingle
       
         Stdsignlermselist.append(rmse2)
        
         Stdsignlefittlist.append(fittingerror2)

         Stdsinglemrelist.append(mresingle)  
 
      
      
      print("The value of eta is")
      print(eta)
                
      RMSEsingle[l]=RMSEsingle[l]/Nbmean                                

      print("The root mean sqaure errors RMSEs are")
                
      print(RMSEsingle[l])                                
                         
      StdsingleRMSE[l]=np.std(np.array(Stdsignlermselist))
          
      print("The standard deviations associated to the RMSEs are")
      
      print(StdsingleRMSE[l])
      
      Fittingsingle[l]=Fittingsingle[l]/Nbmean

      print("The fitting errors FEs are")
 
 
      print(Fittingsingle[l])


      Stdsinglefitting[l]=np.std(np.array(Stdsignlefittlist))

      print("The standard deviations associated to the FEs are")
      
      print(Stdsinglefitting[l])
     
  
      

      MREsingle[l]=MREsingle[l]/Nbmean
     
      print("The mean relative errors MRE are")
 
      print(MREsingle[l])

      
     
      Stdsinglemre[l]=np.std(Stdsinglemrelist)   
     
      print("The standard deviation associated to the MREs are")

      print(Stdsinglemre[l])
     
      pdb.set_trace()

 
    
    return RMSEsingle,StdsingleRMSE,Fittingsingle,Stdsinglefitting,MREsingle,Stdsinglemre


etavalues=[5]#10,15,20,25]
nbepochs=1
Nonnegative=False
step=np.power(10,-5,dtype=float)   #np.power(10,-8,dtype=float)    
max_iter=20
Numberofexamples=9    #6000      #2000   
Minibatchsize=[]                                     
alpha=np.power(10,2,dtype=float)   
theta=np.power(10,-2,dtype=float)   
period=int(max_iter/3)
randomarray=[1,2,3]
trainratio=1/3          #0.80
pool=Pool(3)
RMSEsingle,StdsingleRMSE,Fittingsingle,Stdsinglefitting,MREsingle,Stdsinglemre=ExperimentToyGeneral(etavalues,Nonnegative,Numberofexamples,Minibatchsize,max_iter,step,alpha,theta,nbepochs,randomarray,trainratio,period,pool)
                                                                                                           
