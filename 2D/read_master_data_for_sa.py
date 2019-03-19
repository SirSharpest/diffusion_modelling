#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:29:20 2019

@author: hughesn
"""

from read_data import read_big_json as read_js
from matplotlib.colors import LogNorm
from scipy.spatial.distance import pdist 
import pandas as pd 
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.analyze import morris
import numpy as np

def json_data_to_dataframe(data): 
    data_entries = []
    for k,v in data.items():
        c1, c2 = v.shape
        c1 = c1//2
        c2 = c2//2
        chem_size, _, _, pd_size,  ts = k.rsplit('_')
        for y in range(len(v)):
            for x in range(len(v[y])):
                dist = pdist([[c1,c2], [y,x]], metric='cityblock')[0]
                data_entries.append([v[y,x], chem_size, pd_size, ts, dist])
    return pd.DataFrame(data_entries, columns=['concentration', 'chem_size', 'pd_size', 'ts', 'distance_from_src'])


data = pd.read_json('./data_in_pandas.json')
data['distance_from_src'] = data.apply(lambda x: x['distance_from_src'][0], axis=1) 
# Define the eq
problem = {
  'num_vars': 4,
  'names': ['chem_size', 'pd_size', 'ts', 'distance_from_src'],
  'bounds': [[data['chem_size'].min(), data['chem_size'].max()],
              [data['pd_size'].min(),data['pd_size'].max()],
              [data['ts'].min(),data['ts'].max()],
              [data['distance_from_src'].min(),data['distance_from_src'].max()]]
}

# Run model (example)
Y = np.array(data['concentration'])

# Perform analysis
S= morris.analyze(problem, data.iloc[:,1:].values, data.iloc[:,1].values, print_to_console=True)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# (first and total-order indices with bootstrap confidence intervals)