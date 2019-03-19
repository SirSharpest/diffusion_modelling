#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:30:25 2019

@author: hughesn

https://bionumbers.hms.harvard.edu/bionumber.aspx?id=108250

More useful http://www.jbc.org/content/275/23/17556.full


combine and compare both modules
"""

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def stokes_einstein(x): return ((1.38e-23 * 298.15)/(6*np.pi * 8.9e-4 * x))


sa = stokes_einstein(3.5e-10) * 1e12
auxin = 300  # from Dienum
gfp = 87


def D_eff(D, q, l):
    return (D*q*l)/(D+q*l)


def d_eff():
    problem = {
        'num_vars': 3,
        'names': ['D', 'q', 'l'],
        'bounds': [[gfp//10, sa*10],
                   [0.001, 10],
                   [25, 200]]
    }

    param_values = saltelli.sample(problem, 10000)

    Y = np.array([D_eff(*pv) for pv in param_values])

    return sobol.analyze(problem, Y, print_to_console=True)


def d_analysis(d_eff_on=False):

    if d_eff_on:
        problem = {
            'num_vars': 4,
            'names': ['x', 'D', 'l', 'q'],
            'bounds': [[0, 50],
                       [gfp, sa],
                       [50, 100],
                       [0, 10]]
        }

        def analytical(x, D, q, l):
            D = D_eff(D, q, l)
            t = 60
            return ((1/np.sqrt(4*np.pi*D*t)) * np.exp(- ((np.square(x))/(4*D*t))))
    else:

        problem = {
            'num_vars': 3,
            'names': ['D', 'q', 'l'],
            'bounds': [[gfp//10, sa*10],
                       [0.001, 10],
                       [25, 200]]
        }

        def analytical(x, D, t):
            return ((1/np.sqrt(4*np.pi*D*t)) * np.exp(- ((np.square(x))/(4*D*t))))

    param_values = saltelli.sample(problem, 10000)

    Y = np.array([analytical(*pv) for pv in param_values])

    return sobol.analyze(problem, Y, print_to_console=True)


d_e = d_eff()
d_a = d_analysis(d_eff_on=True)

# Make into DF
pd.concat([i.T for i in d_a.to_df()]).T
