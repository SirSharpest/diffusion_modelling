#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:41:28 2019

@author: nathan
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


def stokes_einstein(x): return ((1.38e-23 * 298.15)/(6*np.pi * 8.9e-4 * x))

def analytical(x, t, D): return (1/np.sqrt(4*np.pi*D*t)
                        * np.exp(- ((np.square(x))/(4*D*t))))

data = pd.read_csv('../kitagawa_estimations.csv')
D = (stokes_einstein(3.5e-10) * 1e+6) 

solutions = [analytical(i/10000, 60*60*14, D) for i in range(-2,3)]
  
plt.plot(solutions)
plt.ylim(0,1)


#data = pd.melt(data, id_vars=['ABA', 'H'])
#g = sns.FacetGrid(data, col='ABA', hue='H', col_wrap=3)
#g.map(plt.plot, 'variable', 'value')

