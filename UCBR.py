# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 02:48:44 2018

@author: priyank
"""


import math
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt


def init(K,T):
    mui = np.random.uniform( low=0.0, high =1.0, size = K)
    rewards = np.zeros((K,2*T))
    for k in range(0,K):
        for t in range(0,2*T):
            rewards[k][t] = sum(np.random.binomial(1,mui[k],size=100))/100
    
    return mui,rewards

def ucbr(K,T,mui,rewards):
#initialise data for all arms for times
    mue=np.zeros(K)
        
    armscount = np.zeros(K)
    maxarm = np.argmax(mui)
    Deltam = 1
    active = np.ones(K)
    m=0
    t=0
    while t<T:
        if sum(active) >1:
            nm = int(np.ceil(2*np.log(T*Deltam*Deltam)/(Deltam*Deltam)))
            cb =np.sqrt(np.log(T*Deltam*Deltam)/(2*nm))
            for k in range(0,K):
                if active[k] == 1:
                    a=int(armscount[k])
                    mue[k]= (mue[k]*armscount[k]+sum(rewards[k][a:nm]))/nm
                    t=t+nm-armscount[k]
                    armscount[k]=nm
        else:
        #display(active)
            remaining = int(np.argmax(active))
            a=int(armscount[remaining])
            end=int(a+T-t)
            mue[remaining] = (mue[remaining]*a+sum(rewards[remaining][a:end]))/(end)
            t=T
            armscount[remaining] = end
            
        maxind = np.argmax(mue)
    
        if sum(active) > 1:
            for k in range(0,K):
                if k!= maxind:
                    if mue[k]+cb < mue[maxind]-cb:
                        active[k]=0
    
        m=m+1
        Deltam=Deltam/2

    total = int(sum(armscount))
    regret = sum(rewards[maxarm][0:total])
    for k in range(0,K):
        a=int(armscount[k])
        regret = regret - sum(rewards[k][0:a])
    return regret
   

regret = np.zeros(19000)

t = 0 
T=100

K=16
mui,rewards = init(K,20000)

while t <19000:
    regret[t]=ucbr(K,T,mui,rewards)
    t=t+1
    T=T+1

plt.plot(regret)
plt.show()
