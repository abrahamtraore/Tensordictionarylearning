#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:22:06 2018

@author: Traoreabraham
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
L=3
Storeaddress='/Users/Traoreabraham/Desktop/OnlineTensorDictionaryLearning/Spatiotemporalpredictionproblem/RMSE'+str(L)
loaded=np.load(Storeaddress+'.npz')
rmseperiteration=loaded['rmsevsiteration']
plt.plot(np.linspace(1,len(rmseperiteration),len(rmseperiteration)),np.array(rmseperiteration), "o-", label="ligne -",color='red',marker="D",markersize=5)
plt.ylabel('RMSE')
plt.xlabel('iteration')
plt.title('RMSE variation per iteration/Training phase'+':'+" "+str('nLag=')+str(L))
plt.savefig('RMSE'+str(L))
pdb.set_trace()


