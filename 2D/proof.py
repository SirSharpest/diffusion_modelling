#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time as ti
from node_diffusion import stokes_einstein, array_to_nodes, diffuse, nodes_to_array, diffuse, D_eff
import numpy as np
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt

sns.set()


def make_numerical(nt):

    # Using moss cell measurements

    dx2 = dx**2
    # Chem radius in meters

    # Set IC and add pulse in centre
    ic = np.zeros((Ys, Xs))
    ic[Ys//2, Xs//2] = 1/dx  # /cell_um

    cur_state = ic

    for i in range(int(nt//dt)):
        cur_state = diffuse(cur_state, dx2, D, q, 0, dt)
    return cur_state


def analytical(x, t, D):
    return ((1/np.sqrt(4*np.pi*D*t)) * np.exp(- ((np.square(x))/(4*D*t))))


r = 3.5e-10
cell_um = 300
num_cells = 5
dx = 10
Xs, Ys = 100, 1
# Chem radius in meters
r = 3.5e-10
# Important model parameters
q = 10
D = stokes_einstein(r) * 1e+12
dt = 0.01
# Being lazy and just updating D once here to be Deff
D = D_eff(D, q, cell_um)


fig = plt.figure(0, figsize=(10, 10))
fig.clf()
fig, ax = plt.subplots(1, sharex=True, num=0)

ts = [1, 10, 30, 60]

num_vals = {}
analytical_vals = {}
t1 = ti.time()
colors = iter(['r', 'g', 'b', 'orange'])
for i in range(0, 4):
    t = ts[i]
    n = Xs*100
    c = next(colors)

    analytical_solution = np.array(
        [analytical(x, t, D)for x in np.linspace(-num_cells//2*cell_um, num_cells//2*cell_um, num=n)])

    numerical_solution = make_numerical(t).ravel()

    num_vals[t] = numerical_solution
    analytical_vals[t] = analytical_solution

    ax.plot(np.linspace(-num_cells//2*cell_um, num_cells//2*cell_um, num=n),
            analytical_solution, label='Analytical: {0}s'.format(t), linewidth=3, c=c, alpha=0.5)

    ax.scatter(np.arange(-Xs//2*dx, +Xs//2*dx, step=dx),
               numerical_solution, label='Numerical: {0}s'.format(t), linewidth=3, c=c, alpha=0.3)
    print(t)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=4, fancybox=True, shadow=True)
ax.set_xlabel(r'$\mu m$')
ax.set_ylabel('C(x,t)')
ax.set_title('')

#ax.set_xticks(np.linspace(-10*cell_um, 10*cell_um, 22, dtype=int))
#labels = np.linspace(-10, 10, 21, dtype=int)
#ax.set_xticklabels(np.linspace(-10, 10, 22, dtype=int))

ax.set_xlim(-300, 300)


fig.tight_layout()
ax.set_yscale('log')
ax.set_ylim(1e-4, 0.03)
fig.canvas.draw()

t2 = ti.time()
print(t2-t1)
plt.show(block=False)
