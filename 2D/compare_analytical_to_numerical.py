#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time as ti
from node_diffusion import stokes_einstein, array_to_nodes, diffuse, nodes_to_array, diffuse, D_eff
import numpy as np
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt

sns.set()
plt.close('all')


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
cell_um = 100
num_cells = 2
dx = 10
Xs, Ys = 300, 1
# Chem radius in meters
r = 3.5e-10
# Important model parameters
q = 10
D = stokes_einstein(r) * 1e+12
dt = 0.001
# Being lazy and just updating D once here to be Deff
D = D_eff(D, q, cell_um)


fig = plt.figure(0, figsize=(10, 10))
fig.clf()
fig, ax = plt.subplots(1, sharex=True, figsize=(5, 5), num=0)

ts = [0.5, 1, 10, 30]

num_vals = {}
analytical_vals = {}
t1 = ti.time()
colors = iter(['r', 'g', 'b', 'orange', 'black'])
for i in range(0, len(ts)):
    t = ts[i]
    n = Xs*1000
    c = next(colors)

    a_start = dx*cell_um
    analytical_solution = np.array(
        [analytical(x, t, D)for x in np.linspace(-a_start, a_start, num=n)])

    numerical_solution = make_numerical(t).ravel()

    num_vals[t] = numerical_solution
    analytical_vals[t] = analytical_solution

    ax.plot(np.linspace(-a_start, a_start, num=n),
            analytical_solution, label='Analytical: {0}s'.format(t), linewidth=4, c=c, alpha=0.5)

    ax.scatter(np.arange(-Xs//2*dx, +Xs//2*dx, step=dx),
               numerical_solution, label='Numerical: {0}s'.format(t), linewidth=4, c=c, alpha=0.3)
    print(t)

ax.set_xlabel(r'$\mu m$')
ax.set_ylabel('C(x,t)')
ax.set_title('')

#ax.set_xticks(np.linspace(-10*cell_um, 10*cell_um, 22, dtype=int))
#labels = np.linspace(-10, 10, 21, dtype=int)
#ax.set_xticklabels(np.linspace(-10, 10, 22, dtype=int))

ax.set_xlim(-400, 400)


ax.set_yscale('log')
ax.set_ylim(1e-4, 0.05)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ] +
             ax.get_xticklabels() + ax.get_yticklabels() + ax.legend().get_texts()):
    item.set_fontsize(20)


ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True, prop={'size': 15})
fig.tight_layout()

t2 = ti.time()
print(t2-t1)
fig.savefig(
    '/Users/hughesn/PHD/Probation/figures/compare_analytical_to_numerical.png', dpi=500)
