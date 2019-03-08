#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:17:55 2019

@author: hughesn
"""

from node_diffusion import stokes_einstein, array_to_nodes, diffuse, nodes_to_array, diffuse, d_eff
import numpy as np
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
sns.set()


def make_numerical(nt):

    # Using moss cell measurements

    dx2 = cell_um**2
    # Chem radius in meters

    # Set IC and add pulse in centre
    ic = np.zeros((Ys, Xs))
    ic[Ys//2, Xs//2] = 1/cell_um

    cur_state = ic

    for i in range(nt):
        cur_state = diffuse(cur_state, dx2, D, q, 0, dt)
    return cur_state


def analytical(x, t, D):
    return ((1/np.sqrt(4*np.pi*D*t)) * np.exp(- ((np.square(x))/(4*D*t))))


t = 60
r = 3.5e-10
cell_um = 50
Xs, Ys = 201, 1
# Chem radius in meters
r = 3.5e-10
# Important model parameters
q = 1
D = stokes_einstein(r) * 1e+12

# Being lazy and just updating D once here to be Deff
D = D_eff(D, q, cell_um)

fig = plt.figure(0)
fig.clf()

fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True, num=0)
ts = np.linspace(10, t, num=4, dtype=int)
ts = [10, 60, 60*2, 60*10]
for i in range(0, 4):
    t = ts[i]
    analytical_solution = np.array([analytical(x, t, D)for x in np.linspace(-cell_um*(Xs//2),
                                                                            cell_um *
                                                                            (Xs//2),
                                                                            num=Xs)])
    if i//2 == 0 and i % 2 == 1:
        ax[i//2, i % 2].plot(np.linspace(-Xs, Xs, num=Xs),
                             analytical_solution, label='analytical', linewidth=3)
        ax[i//2, i % 2].plot(np.linspace(-Xs, Xs, num=Xs),
                             make_numerical(t).ravel(), label='numerical', linestyle='--', linewidth=3)
    else:
        ax[i//2, i % 2].plot(np.linspace(-Xs, Xs, num=Xs),
                             analytical_solution, linewidth=3)
        ax[i//2, i % 2].plot(np.linspace(-Xs, Xs, num=Xs),
                             make_numerical(t).ravel(), linestyle='--', linewidth=3)

    if i//2 > 0:
        ax[i//2, i % 2].set_xlabel('Cell #')
    ax[i//2, i % 2].set_ylabel('C(x,t)')
    ax[i//2, i % 2].set_title('t={0}'.format(t))
    ax[i//2, i % 2].set_xlim(-15, 15)
    ax[i//2, i % 2].set_ylim(1e-5, 1e-1)
    ax[i//2, i % 2].set_yscale('log')
# plt.pause(0.01)
fig.tight_layout()
ax[0, 1].legend()
fig.canvas.draw()
