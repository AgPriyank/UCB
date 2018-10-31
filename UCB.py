# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 02:56:15 2018

@author: priyank
"""

import math
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt

K=20
T=20000

mui = np.random.uniform(low=0.0,high=1.0,size=K)
mue=np.zeros(K)
nit = np.ones(K)
#print('check')
#display(nit[100])
# play once
regret=np.zeros(T)
maxarm = max(mui)
maxind = np.argmax(mui)
for k in range(0,K):
    e = np.random.binomial(1,mui[k],size=500)
    mue[k]=mue[k]+sum(e)/500


ucb=mue+np.sqrt(4*np.log(2)/nit)
regret[1]= maxarm*K-sum(mue)
#display('check1UCB')
#display(ucb)
armscount=np.ones(K)
for t in range(1,T):
    ind = np.argmax(ucb)
    nit[ind]=nit[ind]+1
    e = np.random.binomial(1,mui[ind],size=500)
    #display(sum(e)/5000)
    #display(sum(e)/1000)
    mue[ind] = (armscount[ind]*mue[ind]+sum(e)/500)/(armscount[ind]+1)
    armscount[k]=armscount[k]+1
    ucb = mue+np.sqrt(4*math.log(t)/nit)
    if ind == maxind:
        regret[t]=0
    else:
        regret[t] =  sum(np.random.binomial(1,maxarm,500))/500 -sum(e)/500


#benchmark bounds

benchmark=np.zeros(T)

for t in range(0,T):
    benchmark[t] = 0.1*np.sqrt(K*t*np.log(t))

cumregret=np.cumsum(regret)
plt.plot(cumregret)
plt.show()

#plt.plot(benchmark)
#plt.show()
#display('check2UCB')
#display(ucb)
#display('check1mui')
#display(mui)
#display(mue)
#display(ucb)   
#display(maxarm)
#display(cumregret)