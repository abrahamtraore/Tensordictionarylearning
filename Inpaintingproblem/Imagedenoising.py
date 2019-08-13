#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:02:19 2018

@author: Traoreabraham
"""

from sklearn import linear_model

import matplotlib.pyplot as plt

from multiprocessing import Pool

import sys
#adress="/home/scr/etu/sil821/traorabr/Tensorly"
#sys.path.append(adress)
sys.path.append("..")
backendchoice="numpy"
import tensorly as tl
tl.set_backend(backendchoice)

import scipy.io as sio
from scipy import ndimage
import scipy

from tensorly.backend import mxnet_backend

from PIL import Image 

from MiscellaneousFunctions.OnlineTensor import CyclicBlocCoordinateTucker_setWithPredefinedEpochs

from MiscellaneousFunctions.MethodsTSPen import Tensor_matrixproduct

from MiscellaneousFunctions.MethodsTSPen import Operations_listmatrices


from MiscellaneousFunctions.MethodsTSPen import  Sparse_code#(X,G_init,listoffactors,Nonnegative,step,max_iter,alpha,theta,epsilon):#The parameters are tensors
   
import numpy as np

np.random.seed(1)

import pdb

def Resizeimage(Hyperspectralimg,newwidth,newlength):
    [I,J,K]=np.array(Hyperspectralimg.shape,dtype=int)
    Result=np.zeros((newwidth,newlength,K))
    for k in range(K):
        Result[:,:,k]=scipy.misc.imresize(Hyperspectralimg[:,:,k],(newwidth,newlength),interp='cubic',)
    return Result
        
 
def Droppixels2d(Img,ratio,valdropped):
    [I,J]=np.array(Img.shape,dtype=int)
    Result=np.copy(Img)
    for i in range(int(ratio*I*J)):
        a=np.random.rand()
        b=np.random.rand()
        i=int(I*a)
        j=int(J*b)
        Result[i,j]=valdropped*np.ones(1)    
    return Result       
        
def Droppixels(Img,ratio,valdropped):
    [I,J,K]=np.array(Img.shape,dtype=int)
    Result=np.copy(Img)
    for i in range(int(ratio*I*J)):
        a=np.random.rand()
        b=np.random.rand()
        i=int(I*a)
        j=int(J*b)
        Result[i,j,:]=valdropped*np.ones(K)    
    return Result


def Fix_values_to_one(Newimage,Indexes,val):
    Result=np.copy(Newimage)
    nbindexes=np.shape(Result)[0]
    print(nbindexes)
    for n in range(nbindexes):
        Result[Indexes[n]]=val
    return Result


def Definemasksingleslice2d(Originalimage,val):
    Slice=Originalimage
    Originalimagevector=np.reshape(Slice,np.size(Slice))
    Indexes=np.argwhere(Originalimagevector==val)    
    Newimage=np.zeros(Originalimagevector.shape)
    Newimage[Indexes]=val
    Newimage=np.reshape(Newimage,np.shape(Slice))   
   
    return Newimage


def Definemasksingleslice(Originalimage,val,slicenumber):
    Slice=Originalimage[:,:,slicenumber]
    Originalimagevector=np.reshape(Slice,np.size(Slice))
    Indexes=np.argwhere(Originalimagevector==val)    
    Newimage=np.zeros(Originalimagevector.shape)
    Newimage[Indexes]=val
    Newimage=np.reshape(Newimage,np.shape(Slice))   
   
    return Newimage

def Anytest(matrix,val):
    [N,M]=np.array(matrix.shape)
    result=False
    if(not(np.sum(matrix==val)!=0)):
        result=True
    return result


def Patch_extractionallslices2d(Originalimage,mask,patchsize,val):
    k=0
    [I,J]=np.array(Originalimage.shape,dtype=int)
    n0=np.min([I,J])
    nbpatches=int(np.floor(n0/patchsize))
    Setofpatches=np.zeros((patchsize,patchsize,nbpatches))

    for i in range(nbpatches):
        for j in range(nbpatches):        

           patch=Originalimage[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize]          
           patchmask=mask[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize]
           if not(np.prod(Anytest(patchmask,val))):
             print(patch.shape)
             print(Setofpatches[:,:,k].shape)
             Setofpatches[:,:,k]=patch

    return Setofpatches

def Patch_extractionallslices(Originalimage,patchsize):
    k=0
    [I,J,K]=np.array(Originalimage.shape,dtype=int)
    n0=np.min([I,J])
    nbpatches=int(np.floor(n0/patchsize))
    Setofpatches=np.zeros((patchsize,patchsize,K,nbpatches))

    for i in range(nbpatches):
        for j in range(nbpatches):        
           patch=Originalimage[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize,:]
           Setofpatches[:,:,:,k]=patch
    return Setofpatches

def Repetition(matrix,nbcopies):
    [N,M]=np.array(matrix.shape,dtype=int)
    result=np.zeros((N,M,nbcopies))
    for n in range(nbcopies):
       result[:,:,n]=matrix
    return result


def Numberofnonmaskedpixels(Ind):
    Columnofinterest=Ind[:,2]
    Indexesofinterest=np.where(Columnofinterest==0)
    result=len(Indexesofinterest[0])
    return result
    
def Patchconsiderationdictionary(Dictionarymatrices,Ind):
    M=np.array(Ind.shape,dtype=int)[0]
    Firstindexes=np.zeros(M)
    Secondindexes=np.zeros(M)
    Thirdindexes=np.zeros(M)    
    for m in range(M):
        Firstindexes[m]=np.array(Ind[m,:][0],dtype=int)
        Secondindexes[m]=np.array(Ind[m,:][1],dtype=int)
        Thirdindexes[m]=np.array(Ind[m,:][2],dtype=int)
    Firstindexes=np.array(np.unique(Firstindexes),dtype=int)
    Secondindexes=np.array(np.unique(Secondindexes),dtype=int)
    Thirdindexes=np.array(np.unique(Thirdindexes),dtype=int)
    A1tilde=Dictionarymatrices[0][Firstindexes,:]
    A2tilde=Dictionarymatrices[1][Secondindexes,:]
    A3tilde=Dictionarymatrices[2][Thirdindexes,:]
    Dictionarymatricesnomask=[A1tilde,A2tilde,A3tilde]
    return Dictionarymatricesnomask

def Reorder(Hyperspectralimage):
    [K,W,H]=np.array(Hyperspectralimage.shape,dtype=int)
    result=np.zeros((W,H,K))
    for k in range(K):
        result[:,:,k]=Hyperspectralimage[k,:,:]
    return result


def Reconstruction(Originalimage,patchsize,Coretensorsize,alpha,theta,pool):
    
    [I,J,K]=np.array(Originalimage.shape,dtype=int)
    Listrestauration=[]
    n0=np.min([I,J])
    Imrestored=np.zeros((I,J,K))
    nbpatches=int(np.floor(n0/patchsize)) 
    Setofpatches=np.zeros((patchsize,patchsize,3,nbpatches))
  
    Setofpatches=Patch_extractionallslices(Originalimage,patchsize)
    Xtrain_set=[]
    for l in range(nbpatches):
        Xtrain_set.append(tl.tensor(Setofpatches[:,:,:,l]))
    max_iter=5
    period=20
    Nonnegative=False
 
    epsilon=np.power(10,-3,dtype=float)
    step=np.power(10,-5,dtype=float)
    Setting="Single"
    nbepochs=1
    Reprojectornot=False
    Minibatchsize=[]
    Pre_existingfactors=[]

    Pre_existingG_settrain=[]
    
    for l in range(nbpatches):
        
        Pre_existingG_settrain.append(tl.tensor(np.maximum(np.random.normal(loc=0,scale=1,size=Coretensorsize),0)))

    Pre_existingfactors=[tl.tensor(np.maximum(np.random.normal(loc=0,scale=1,size=(patchsize,Coretensorsize[0])),0)),tl.tensor(np.maximum(np.random.normal(loc=0,scale=1/4,size=(patchsize,Coretensorsize[1])),0)),tl.tensor(np.maximum(np.random.normal(loc=0,scale=1/4,size=(K,Coretensorsize[2])),0))]
    
    Pre_existingP=[tl.tensor(np.random.normal(loc=0,scale=1,size=(patchsize,Coretensorsize[0]))),tl.tensor(np.random.normal(loc=0,scale=1,size=(patchsize,Coretensorsize[1]))),tl.tensor(np.random.normal(loc=0,scale=1,size=(K,Coretensorsize[2])))]
    
    Pre_existingQ=[tl.tensor(np.random.normal(loc=0,scale=1,size=(Coretensorsize[0],Coretensorsize[0]))),tl.tensor(np.random.normal(loc=0,scale=2,size=(Coretensorsize[1],Coretensorsize[1]))),tl.tensor(np.random.normal(loc=0,scale=1,size=(Coretensorsize[2],Coretensorsize[2])))]
         
    Dictionarymatrices,listobjectivefunctionvalues,Objectivefunction_per_epoch=CyclicBlocCoordinateTucker_setWithPredefinedEpochs(Xtrain_set,Coretensorsize,Pre_existingfactors,Pre_existingG_settrain,backendchoice,Pre_existingP,Pre_existingQ,Nonnegative,Reprojectornot,Setting,Minibatchsize,step,alpha,theta,max_iter,epsilon,period,nbepochs,pool)
    
    #Dm=list(Dictionarymatrices)
    
    #Dictionarymatricesconverted=Operations_listmatrices(Dictionarymatrices,"Arrayconversion")
    #pdb.set_trace()
    
    Nonnegative=False
    for i in range(nbpatches):
        for j in range(nbpatches):
            
            Noisypatch=Originalimage[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize,:]  
            
            G_init=tl.tensor(np.random.normal(loc=0,scale=1,size=Coretensorsize))
            
            Activationcoeff=Sparse_code(Noisypatch,G_init,Dictionarymatrices,Nonnegative,step,max_iter,alpha,theta,epsilon)[0]
            
            Restore=Tensor_matrixproduct(Activationcoeff,Dictionarymatrices)
            
            Imrestored[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize,:]=Restore
            #pdb.set_trace()
            #Aux=np.resize(Aux,np.size(Aux))        
            #patchmask=Repetition(patchmask,K)
            #Auxmask=np.resize(patchmask,np.size(patchmask))
            #Ind=np.where(Auxmask!=val)[0]      #np.nonzero(Auxmask)[0]
            
            #yy=Aux#[Ind]
            #Dictionarymatrix=np.kron(Dictionarymatricesconverted[0],Dictionarymatricesconverted[1])
            #Dictionarymatrix=np.kron(Dictionarymatrix,Dictionarymatricesconverted[2])
            #Dma=Dictionarymatrix
            #pdb.set_trace()
            
            #clf=linear_model.Lasso(alpha=penaltylasso,fit_intercept=False,positive=False)#fit_intercept=False
            
            #clf.fit(Dma,yy)

            #Activationcoeff=np.reshape(clf.coef_,Coretensorsize)
            
            #Restore=Tensor_matrixproduct(Activationcoeff,Dm)
            #Restore=mxnet_backend.to_numpy(Restore)

            Listrestauration.append(Restore)

    return Imrestored,Listrestauration    

from astropy.io import fits
hdulist=fits.open('DATACUBE_A2744_v1.0.fits')    
#Good candidate: 'sc_4_DATACUBEFINALuser_20140216T010259_cf767044.fits'


image_data=hdulist[1].data
Hyperspectalimage=np.nan_to_num(image_data) #[0:200,:,:]
Hyperspectalimage=Reorder(Hyperspectalimage)
[W,H,K]=np.array(Hyperspectalimage.shape,dtype=int)
Hyperspectralimagenoisy=Hyperspectalimage+np.random.normal(loc=0,scale=5,size=(W,H,K))
#pdb.set_trace()
patchsize=8
Rank=8
alpha=10
theta=1
[width,length,spectralbands]=np.array(Hyperspectalimage.shape,dtype=int)
Corentensorsize=[int(Rank),int(Rank),int(Rank)]
pool=Pool(20)
Imrestored,Listrestauration=Reconstruction(Hyperspectralimagenoisy,patchsize,Corentensorsize,alpha,theta,pool)
plt.imshow(Imrestored[:,:,0])
plt.show()
pdb.set_trace()
            
            
            
