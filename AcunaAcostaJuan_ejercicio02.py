#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# This repo contains an introduction to [Jupyter](https://jupyter.org) and [IPython](https://ipython.org).
# 
# Outline of some basics:
# 
# * [Notebook Basics](../examples/Notebook/Notebook%20Basics.ipynb)
# * [IPython - beyond plain python](../examples/IPython%20Kernel/Beyond%20Plain%20Python.ipynb)
# * [Markdown Cells](../examples/Notebook/Working%20With%20Markdown%20Cells.ipynb)
# * [Rich Display System](../examples/IPython%20Kernel/Rich%20Output.ipynb)
# * [Custom Display logic](../examples/IPython%20Kernel/Custom%20Display%20Logic.ipynb)
# * [Running a Secure Public Notebook Server](../examples/Notebook/Running%20the%20Notebook%20Server.ipynb#Securing-the-notebook-server)
# * [How Jupyter works](../examples/Notebook/Multiple%20Languages%2C%20Frontends.ipynb) to run code in different languages.

# You can also get this tutorial and run it on your laptop:
# 
#     git clone https://github.com/ipython/ipython-in-depth
# 
# Install IPython and Jupyter:
# 
# with [conda](https://www.anaconda.com/download):
# 
#     conda install ipython jupyter
# 
# with pip:
# 
#     # first, always upgrade pip!
#     pip install --upgrade pip
#     pip install --upgrade ipython jupyter
# 
# Start the notebook in the tutorial directory:
# 
#     cd ipython-in-depth
#     jupyter notebook

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

datos=np.genfromtxt("numeros_20.txt")
datosTr=datos[0:10,:]
datosTe=datos[10:,:]


def coeficientes(a,m):
    m+=1
    d=np.zeros((len(a),m))
    for i in range(m):
        d[:,i]=np.power(a[:,0],i)
        
    pi=np.linalg.pinv(d)
    beta=np.dot(pi,a[:,1])
    return beta


def polifit(beta,x):
    y=np.zeros(len(x))
    for i in range(len(beta)):
        y+=beta[i]*x**i
    return y



M=[0,1,3,9]
plt.figure()
for i in range(4):
    x=np.linspace(np.min(datosTr[:,0]),np.max(datosTr[:,0]) , 100)
    y=polifit(coeficientes(datosTr,M[i]),x)
    plt.subplot(2,2,i+1)
    plt.scatter(datosTr[:,0],datosTr[:,1])
    plt.plot(x,y,"r")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("M={}".format(M[i]))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.savefig("polifit.png", bbox_inches='tight')

def E(y,b):
    return np.sum((y-b)**2)

"""
Etr=np.zeros(10)
Ete=np.zeros(10)
a=[0,1,2,3,4,5,6,7,8,9]
x=np.linspace(np.min(datosTr[:,0]),np.max(datosTr[:,0]) , 10)

for i in range(10):
    Etr[i]=np.sqrt(E(polifit(coeficientes(datosTr,i),x),datosTr[:,1])/10)
    Ete[i]=np.sqrt(E(polifit(coeficientes(datosTr,i),x),datosTe[:,1])/10)
"""
Etr=[]
Ete=[]
a=[0,1,2,3,4,5,6,7,8,9]
x=np.linspace(np.min(datosTr[:,0]),np.max(datosTr[:,0]) , 10)
for i in range(10):
    Etr.append(E(polifit(coeficientes(datosTr,a[i]),datosTr[:,0]),datosTr[:,1])/10.)
    Ete.append(E(polifit(coeficientes(datosTe,a[i]),datosTr[:,0]),datosTe[:,1])/10.)



plt.figure()
plt.plot(a,np.log(Etr),'.--',label="Training")
plt.plot(a,np.log(Ete),'.r--',label="Test")
plt.legend()
plt.ylabel("E_rms")
plt.xlabel("M")
plt.savefig("Erms.png", bbox_inches='tight')

