#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
"""
Created on Wed Feb 13 14:44:46 2019

Here we explore diffusion through connectivity between nodes
We assume intra-cellular diffusion behaves as expected by fickian laws

@author: nathan
"""


def stokes_einstein(x):
    return ((1.38e-23 * 298.15) / (6 * np.pi * 8.9e-4 * x))


def diffuse(nodes, dx2, dy2, D, ts, pd_rate, b):
    """
    nodes a np array of nodes
    dx2, dx2 the difference in x,y squared
    D diffusion constant
    f a method to make a diffusion matrix of available cells

    returns a generator for u
    """
    Y, X = nodes.shape
    un = nodes_to_array(nodes).copy()
    u = un.copy()
    conn = connectivity(un, pd_rate)
    beta = production(un, b)
    for y in range(0, Y):
        for x in range(0, X):
            x1 = (un[y, x-1] * conn[y, x-1] if x > 0 else 0)
            x2 = (un[y, x+1] * conn[y, x+1] if x < X-1 else 0)
            y1 = (un[y-1, x] * conn[y-1, x] if y > 0 else 0)
            y2 = (un[y+1, x] * conn[y+1, x] if y < Y-1 else 0)
            u[y, x] = d(u[y, x], y1, y2, x1, x2, D, dx2, dy2, beta[y, x])
    array_update_nodes(u, nodes)
    return nodes


def production(c, b):
    return np.zeros(c.shape)  # + (b * np.where)


def d(c0, y1, y2, x1, x2, D, dx2, dy2, b):
    c = c0 + D * ((x1 - 2*c0 + x2)/dx2 + (y1 - 2*c0 + y2)/dy2) + b
    return c


def pd_perm(x, pd_rate=0.8):
    return x*pd_rate


def connectivity(c, pd_rate):
    """
    Returns a connectivity mask for the current cell array
    """
    return pd_perm(np.ones(c.shape), pd_rate=pd_rate)


def to_dataframe(A):
    x1 = np.repeat(np.arange(A.shape[0]), len(
        A.flatten())/len(np.arange(A.shape[0])))
    x2 = np.tile(np.arange(A.shape[1]), int(
        len(A.flatten())/len(np.arange(A.shape[1]))))
    x3 = A.flatten()

    # TODO: Add actual numbers here
    m = np.array([1 for i in range(0, len(x3))])
    # m[3:] = 0
    return pd.DataFrame(np.array([x1, x2, x3, m]).T, columns=['X', 'Y', 'C', 'M'])


def nodes_to_array(nodes):
    Y, X = nodes.shape
    arr = []
    for y in range(0, Y):
        y_arr = []
        for x in range(0, X):
            y_arr.append(nodes[y, x].get_c())
        arr.append(y_arr)
    return np.array(arr)


def array_update_nodes(arr, nodes):
    Y, X = nodes.shape
    for y in range(0, Y):
        for x in range(0, X):
            nodes[y, x].update_c(arr[y, x])


def array_to_nodes(arr):
    Y, X = arr.shape
    return np.array([[Node(x, y, arr[y, x]) for x in range(0, X)] for y in range(0, Y)])


def get_closed_nodes(nodes):
    pass


class Node:
    """
    Is just a holder of data for each node
    """

    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.c = c
        self.closed = False

    def update_c(self, c):
        self.c = c

    def get_c(self):
        return self.c

    def set_closed(self):
        self.closed = True


if __name__ == '__main__':
    t1 = time.time()
    # Number of time points, in seconds
    nt = 60*60*4
    # delta t is just one
    dt = 1

    # Number of cells per direction
    Xs, Ys = 25, 25

    # Using moss cell measurements
    cell_um = 100

    dx2, dy2 = cell_um**2, cell_um**2

    # Chem radius in meters
    r = 3.5e-10

    # Speed in micrometers
    D = stokes_einstein(r) * 1e+12

    # Set IC and add pulse in centre
    ic = np.zeros((Ys, Xs))
    ic[1] = 1

    # Make nodes from IC
    nodes = array_to_nodes(ic)

    # some cell and PD constants
    pd_rate = 0.9
    beta = .01
    prod_upper_lim = 0.6
    prod_lower_lim = 0.1

    # Get results from cells
    Cells = [deepcopy(diffuse(nodes, dx2, dy2, D, i, pd_rate, beta))
             for i in range(1, nt)]

    print('{0} took {1:.2f} seconds  to simulate'.format(nt, time.time()-t1))

    # sns.scatterplot(data=to_dataframe(nodes_to_array(Cells[1])),
    #                 x='X', y='Y', size='C', hue='C',
    #                 legend=False, sizes=(300, 300), style='M', markers=['s', 'X'])

    # plt.show()
