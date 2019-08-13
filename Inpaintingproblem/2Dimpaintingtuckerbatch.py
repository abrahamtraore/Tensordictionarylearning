#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 09:23:14 2018

@author: Traoreabraham
"""

from sklearn import linear_model


from multiprocessing import Pool

import time

import sys
#adress="/home/scr/etu/sil821/traorabr/Tensorly"
#sys.path.append(adress)
sys.path.append("..")
sys.path.append("/home/scr/etu/sil821/traorabr/Tensorly/")
backendchoice="numpy"
import tensorly as tl
tl.set_backend(backendchoice)

import scipy.io as sio
from scipy import ndimage
import scipy

sys.path.append("/home/scr/etu/sil821/traorabr/OnlineTensorDictionaryLearning/")

from tensorly.backend import mxnet_backend

from PIL import Image 

from MiscellaneousFunctions.OnlineTensor import CyclicBlocCoordinateTucker_setWithPredefinedEpochs

from MiscellaneousFunctions.MethodsTSPen import Tensor_matrixproduct

from MiscellaneousFunctions.MethodsTSPen import Operations_listmatrices

from MiscellaneousFunctions.TuckerBatch import TuckerBatch


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

#Originalimage=np.array(Image.open("parrot_original_mask2.png"))
#slicenumber=0
#Newimage=Definemasksingleslice(Originalimage,slicenumber)
#plt.imshow(Newimage)
#plt.show()
    
#def Anytest(matrix,val):
#    [N,M]=np.array(matrix.shape)
#    result=np.zeros(M)
#    for m in range(M):
#        indicator=np.mean(matrix[:,m]==val)
#        result[m]=0
#     
#        if(indicator==1):
#          result[m]=1
#    result=np.array(result,dtype=int)
#    return result

def Anytest(matrix,val):
    [N,M]=np.array(matrix.shape)
    result=False
    if(not(np.sum(matrix==val)!=0)):
        result=True
    return result
#matrix=np.array([[0,0,3],[0,0,3],[0,0,3]])
#result=Anytest(matrix)

def Patch_extractionallslices2d(Originalimage,mask,patchsize,val):
    k=0
    [I,J]=np.array(Originalimage.shape,dtype=int)
    n0=np.min([I,J])
    nbpatches=int(np.floor(n0/patchsize))
    Setofpatches=np.zeros((patchsize,patchsize,nbpatches))
#    while (k<nbpatches):
    for i in range(nbpatches):
        for j in range(nbpatches):        
        #a=np.random.rand()
        #b=np.random.rand()
        #print("The values of a and b are")
        #print(a)
        #print(b)
        #x=int(np.floor(a*(n0-patchsize)))
        #y=int(np.floor(b*(n0-patchsize)))
        
           patch=Originalimage[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize]          #Originalimage[x:x+patchsize,y:y+patchsize,:]
           patchmask=mask[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize]         #mask[x:x+patchsize,y:y+patchsize]
           if not(np.prod(Anytest(patchmask,val))):
             
             Setofpatches[:,:,k]=patch
    

#    while (k<nbpatches):
#        
#        a=np.random.rand()
#        b=np.random.rand()
#        #print("The values of a and b are")
#        #print(a)
#        #print(b)
#        x=int(np.floor(a*(n0-patchsize)))
#        y=int(np.floor(b*(n0-patchsize)))
#        
#        patch=Originalimage[x:x+patchsize,y:y+patchsize,:]
#        patchmask=mask[x:x+patchsize,y:y+patchsize]
#        #plt.imshow(patch)
#        #plt.show()
#        #pdb.set_trace()
#        if not(np.prod(Anytest(patchmask))):
#            #print(k)
#            Setofpatches[:,:,:,k]=patch
#            k=k+1            
    return Setofpatches





def Patch_extractionallslices(Originalimage,mask,patchsize,val):
    k=0
    [I,J,K]=np.array(Originalimage.shape,dtype=int)
    n0=np.min([I,J])
    nbpatches=int(np.floor(n0/patchsize))
    Setofpatches=np.zeros((patchsize,patchsize,K,nbpatches))
#    while (k<nbpatches):
    for i in range(nbpatches):
        for j in range(nbpatches):        
        #a=np.random.rand()
        #b=np.random.rand()
        #print("The values of a and b are")
        #print(a)
        #print(b)
        #x=int(np.floor(a*(n0-patchsize)))
        #y=int(np.floor(b*(n0-patchsize)))
        
           patch=Originalimage[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize,:]          #Originalimage[x:x+patchsize,y:y+patchsize,:]
           patchmask=mask[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize]         #mask[x:x+patchsize,y:y+patchsize]
           if not(np.prod(Anytest(patchmask,val))):
             
             Setofpatches[:,:,:,k]=patch
    

#    while (k<nbpatches):
#        
#        a=np.random.rand()
#        b=np.random.rand()
#        #print("The values of a and b are")
#        #print(a)
#        #print(b)
#        x=int(np.floor(a*(n0-patchsize)))
#        y=int(np.floor(b*(n0-patchsize)))
#        
#        patch=Originalimage[x:x+patchsize,y:y+patchsize,:]
#        patchmask=mask[x:x+patchsize,y:y+patchsize]
#        #plt.imshow(patch)
#        #plt.show()
#        #pdb.set_trace()
#        if not(np.prod(Anytest(patchmask))):
#            #print(k)
#            Setofpatches[:,:,:,k]=patch
#            k=k+1            
    return Setofpatches

#Originalimage=np.array(Image.open("parrot_original_mask2.png"))
#slicenumber=0
#mask=Definemasksingleslice(Originalimage,slicenumber)
#patchsize=18
#nbpatches=50
#Setofpatches=Patch_extractionallslices(Originalimage,mask,nbpatches,patchsize)
#Setofpatches=tl.tensor(Setofpatches)
#Listofpatches=[]
#for l in range(nbpatches):
#    Listofpatches.append(Setofpatches[:,:,:,l])
    
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

#def Patchconsiderationdata(Aux,Ind):
#    M=np.array(Ind.shape,dtype=int)[0]
#    numberoflastdimensions=np.unique(Ind[:,2])#spectral bands numerotation
#    nbnomaskpixels=Numberofnonmaskedpixels(Ind)
#    patchwithoutmask=np.zeros((nbnomaskpixels,nbnomaskpixels,numberoflastdimensions))
#    
#    
    
    
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
      

def Reconstruction(Originalimage,patchsize,Rank,alpha,theta,val,pool):
    
    [I,J]=np.array(Originalimage.shape,dtype=int)
    Listrestauration=[]
    n0=np.min([I,J])
    Imrestored=np.zeros((I,J))
    #nbx=int(np.floor(n0/patchsize))
    nbpatches=int(np.floor(n0/patchsize)) 
    Setofpatches=np.zeros((patchsize,patchsize,nbpatches))
    mask=Definemasksingleslice2d(Originalimage,val)
    Setofpatches=Patch_extractionallslices2d(Originalimage,mask,patchsize,val)
    Xtrain=np.zeros((nbpatches,patchsize,patchsize))
    for l in range(nbpatches):
        Xtrain[l,:,:]=tl.tensor(Setofpatches[:,:,l])
    max_iter=500#0
    period=3
    Nonnegative=False
 
    epsilon=np.power(10,-3,dtype=float)
    step=np.power(10,-5,dtype=float)#np.power(10,-5,dtype=float)
    Setting="Single"
    nbepochs=3
    Reprojectornot=False
    Minibatchsize=[]
    Pre_existingfactors=[]
    Coretensorsize=[Rank,Rank]
    penaltylasso=alpha*theta
    Ginittrain=np.zeros((nbpatches,Coretensorsize[0],Coretensorsize[1]))
    
    for l in range(nbpatches):
        
        Ginittrain[l,:,:]=tl.tensor(np.maximum(np.random.normal(loc=0,scale=1,size=Coretensorsize),0))

    
    listoffactorsinit=[tl.tensor(np.identity(nbpatches)),tl.tensor(np.maximum(np.random.normal(loc=0,scale=1,size=(patchsize,Coretensorsize[0])),0)),tl.tensor(np.maximum(np.random.normal(loc=0,scale=1,size=(patchsize,Coretensorsize[1])),0))]
    
    start_timetraining=time.clock()
    
    Dictionarymatrices,errorlist,listobjectivefunctionvalues,nbiter=TuckerBatch(Xtrain,[nbpatches,Coretensorsize[0],Coretensorsize[1]],max_iter,listoffactorsinit,Ginittrain,Nonnegative,backendchoice,Reprojectornot,alpha,theta,step,epsilon,pool)                                        
    
    #Dictionarymatrices,errorlist,listobjectivefunctionlist,Objectivefnction_per_epoch,Time_per_epoch,nbiter=TuckerBatch(Xtrain,[nbpatches,Coretensorsize[0],Coretensorsize[1]],max_iter,listoffactorsinit,Ginittrain,Nonnegative,backendchoice,Reprojectornot,alpha,theta,step,epsilon,pool)
    #pdb.set_trace()
    
    
    end_timetraining=time.clock()
    Runningtime=end_timetraining-start_timetraining
    print("The running time")
    print(Runningtime)
    pdb.set_trace()
    #adress='/Users/Traoreabraham/Desktop/OnlineTensorDictionaryLearning/Hyperspecimpainting/Objectivefunctions/TuckerObjectivefunction'+str(Rank)   
    
    #adress='/home/scr/etu/sil821/traorabr/ImageInpainting/Objectivefunctions/TuckerObjectivefunction'+str(Rank)
    #np.savez_compressed(adress,Objectivefunction=listobjectivefunctionlist,Objectivefnctionperepoch=Objectivefnction_per_epoch,Timeperepoch=Time_per_epoch)    
    #pdb.set_trace()
    
    
    Dm=list(Dictionarymatrices[1:3])
    
    Dictionarymatricesconverted=Operations_listmatrices(Dictionarymatrices[1:3],"Arrayconversion")
    
    mask=Definemasksingleslice2d(Originalimage,val)
    #plt.imshow(mask)
    #plt.show()
    #pdb.set_trace()
    for i in range(nbpatches):
        for j in range(nbpatches):
            #indx=np.array(np.linspace(i*patchsize+1,(i+1)*patchsize,patchsize),dtype=int)
            #indy=np.array(np.linspace(j*patchsize+1,(j+1)*patchsize,patchsize),dtype=int)
            
            
            #pdb.set_trace()
            #patchmask=mask[i*patchsize+1:(i+1)*patchsize,j*patchsize+1:(j+1)*patchsize]#[indx,indy]         
            #Aux=Originalimage[i*patchsize+1:(i+1)*patchsize,j*patchsize+1:(j+1)*patchsize,:]#[indx,indy,:]            
            patchmask=mask[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize]#[indx,indy]         
            #plt.imshow(patchmask)
            #plt.show()
            #print(patchmask)
            
            Aux=Originalimage[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize]#[indx,indy,:]            
            Aux=np.resize(Aux,np.size(Aux))        
            #patchmask=Repetition(patchmask,K)
            Auxmask=np.resize(patchmask,np.size(patchmask))
            Ind=np.where(Auxmask!=val)[0]      #np.nonzero(Auxmask)[0]
            
            yy=Aux[Ind]
            Dictionarymatrix=np.kron(Dictionarymatricesconverted[0],Dictionarymatricesconverted[1])
            Dma=Dictionarymatrix[Ind,:]
            clf=linear_model.Lasso(alpha=penaltylasso,fit_intercept=False,positive=True)#fit_intercept=False
            #print(Dma.shape)
            #print(yy.shape)
            clf.fit(Dma,yy)
            #print(clf.coef_.shape)
            #pdb.set_trace()
         
            Activationcoeff=np.reshape(clf.coef_,(Rank,Rank))
            
            Restore=Tensor_matrixproduct(Activationcoeff,Dm)
            Restore=mxnet_backend.to_numpy(Restore)
            #plt.imshow(Originalimage[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize,:])
            #plt.show()
            #plt.imshow(Restore)
            #plt.show()
            
            Listrestauration.append(Restore)
            #plt.imshow(Restore)
            #plt.show()
            #plt.imshow(Originalimage[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize,:])
            #plt.show()
            #pdb.set_trace()
            #print(np.argwhere(Restore==0))
            #print(i*patchsize)
            #print((i+1)*patchsize)
            #print(j*patchsize)
            #print((j+1)*patchsize)
            Imrestored[i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize]=Restore
            #print(Restore)            
            #pdb.set_trace()
            #print("The values of i and j are")
            #print(i,j)
    
    return Imrestored,Listrestauration,Runningtime


#Image en noir et blanc
val=255
ratio=0.5

Hyperspectralimg=np.array(Image.open("Barbaraimg.jpg").convert('L'))
#np.array(Image.open("/Users/Traoreabraham/Desktop/OnlineTensorDictionaryLearning/Inpaintingproblem/Barbaraimg.jpg").convert('L'))   #scipy.misc.imread("Barbara.png")#np.array(Image.open("Barbara.png"))
#newwidth=256
#newlength=256
#Hyperspectralimg=scipy.misc.imresize(Hyperspectralimg,(newwidth,newlength),interp='cubic')
#plt.imshow(Hyperspectralimg)
#plt.show()
#pdb.set_trace()
[W,H]=np.array(Hyperspectralimg.shape,dtype=int)
Hyperspectralimgpixelsdropped=Droppixels2d(Hyperspectralimg,ratio,val)
##pdb.set_trace()
patchsize=8
Rank=18
alpha=0.001 
theta=1
#Corentensorsize=[int(Rank),int(Rank)]
pool=Pool(20)
Imrestored,Listrestauration,Runningtime=Reconstruction(Hyperspectralimgpixelsdropped,patchsize,Rank,alpha,theta,val,pool)

                                                    
#plt.imshow(Imrestored)
#plt.show()
#plt.imshow(Hyperspectralimgpixelsdropped)
#plt.show()
MSE=(np.linalg.norm(Imrestored-Hyperspectralimg)**2)/(W*H)
PSNR=(20/np.log(10))*np.log(255/np.sqrt(MSE))           
print("The running time is")
print(Runningtime)
print("The PSNR is")
print(PSNR)


pdb.set_trace()   
