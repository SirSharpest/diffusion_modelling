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
            'names': ['D', 'x', 'l', 'q'],
            'bounds': [[gfp, sa],
                       [0, 50],
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


def prep_data_frames(d_result):

    tbl_a = pd.concat([i.T for i in d_result.to_df()[:-1]]).T

    tbl_2nd = d_result.to_df()[-1].T
    tbl_a = tbl_a.T

    tbl_a = tbl_a[::2]
    tbl_2nd = tbl_2nd[:1]

    tbl_a_melted = pd.melt(tbl_a.reset_index(), id_vars='index',
                           value_name='Si', var_name='variable')

    tbl_2nd_melted = pd.melt(tbl_2nd.reset_index(), id_vars='index',
                             value_name='Si', var_name='variable')

    tbl_a_melted['variable'] = tbl_a_melted['variable'].map(''.join)
    tbl_2nd_melted['variable'] = tbl_2nd_melted['variable'].map(','.join)
    tbl_2nd_melted['variable'] = ['({0})'.format(a)
                                  for a in tbl_2nd_melted['variable']]
    tbl_2nd_melted['Si'] = abs(tbl_2nd_melted['Si'])

    return (tbl_a_melted, tbl_2nd_melted)


sa_d_eff = d_eff()
sa_d_with_d_eff = d_analysis(d_eff_on=True)
sa_d = d_analysis(d_eff_on=False)


fig = plt.figure(0, figsize=(5, 5))
fig.clf()
fig, ax = plt.subplots(2, 2, num=0, sharey=True)
[a.cla() for a in ax.ravel()]


s1, s2 = prep_data_frames(sa_d_eff)
sns.catplot(x='variable', y='Si', hue='index',
            data=s1, kind='bar', ax=ax[0, 0])
sns.catplot(x='variable', y='Si', hue='index',
            data=s2, kind='bar', ax=ax[0, 1])

ax[0, 0].set_title(r'Effect on diffusion by using $D$')
ax[0, 1].set_title(r'Effect on diffusion by using $D$')

s3, s4 = prep_data_frames(sa_d_with_d_eff)
sns.catplot(x='variable', y='Si', hue='index',
            data=s3, kind='bar', ax=ax[1, 0])
sns.catplot(x='variable', y='Si', hue='index',
            data=s4, kind='bar', ax=ax[1, 1])

ax[1, 0].set_title(r'Effect on diffusion by using $D_{eff}$')
ax[1, 1].set_title(r'Effect on diffusion by using $D_{eff}$')

fig.tight_layout()
fig.canvas.draw()
