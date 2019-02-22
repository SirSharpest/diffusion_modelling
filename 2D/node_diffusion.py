#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.animation as animation
import networkx as nx
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
            u[y, x] = d(u[y, x], conn[y, x], y1, y2,
                        x1, x2, D, dx2, dy2, beta[y, x])
    array_update_nodes(u, nodes)
    return nodes


def production(c, b):
    return np.zeros(c.shape)  # + (b * np.where)


def d(c0, c0_pd, y1, y2, x1, x2, D, dx2, dy2, b):
    c = c0 + D * ((x1 - (2*c0*c0_pd) + x2)/dx2 +
                  (y1 - (2*c0*c0_pd) + y2)/dy2) + b
    return c


def pd_perm(x, pd_rate=0):
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
    df = pd.DataFrame(np.array([x1, x2, x3, m]).T,
                      columns=['X', 'Y', 'C', 'M'])
    df['norm_C'] = np.log(df['C'])
    return df


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


def draw_as_network(nodes, ax, draw_labels=False, title=''):
    G = nx.Graph()
    Y, X = nodes.shape
    ax.grid(False)
    arr = nodes_to_array(nodes)
    sizes = np.zeros(nodes.shape)
    labels = {}
    log_scale = 1024
    cut_off_of_interest = 1e-4
    pos = {}
    for y in range(0, Y):
        for x in range(0, X):
            cur_node = (y*X) + x
            if x < X-1:
                G.add_edge((cur_node), (cur_node+1),
                           weight=1 if arr[y, x] > cut_off_of_interest else 0)
            if y < Y-1:
                G.add_edge((cur_node), (cur_node+X),
                           weight=1 if arr[y, x] > cut_off_of_interest else 0)

            pos[cur_node] = np.array([x, y])
            sizes[y, x] = arr[y, x]
            lbl = arr[y, x]

            labels[cur_node] = "{0:1.2e}".format(
                np.around(lbl, 4)) if lbl > cut_off_of_interest else ''

    with np.errstate(divide='ignore'):
        sizes = np.log2(sizes.ravel()*log_scale)*75

    new_sizes = sizes.copy()

    for idx, i in enumerate(G.nodes):
        new_sizes[idx] = sizes[i]

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=new_sizes, ax=ax)

    # Decide nodes
    # e_on = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
    # e_off = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 0]

    nx.draw_networkx_edges(G, pos,
                           width=1, edge_color='b', ax=ax)

    if draw_labels:
        nx.draw_networkx_labels(G, pos, font_size=7,
                                font_family='sans-serif', labels=labels, ax=ax)

    ax.set_xlim(-1, X)
    ax.set_ylim(-1, Y)
    ax.set_title(title)


if __name__ == '__main__':
    sns.set()
    t1 = time.time()
    # Number of time points, in seconds
    nt = 60*60*2
    # delta t is just one
    dt = 1

    # Number of cells per direction
    Xs, Ys = 5, 1

    # Using moss cell measurements
    cell_um = 100

    dx2, dy2 = cell_um**2, cell_um**2

    # Chem radius in meters
    r = 3.5e-10

    # Speed in micrometers
    D = stokes_einstein(r) * 1e+12

    # Set IC and add pulse in centre
    ic = np.zeros((Ys, Xs))
    ic[Ys//2, Xs//2] = 1

    # Make nodes from IC
    nodes = array_to_nodes(ic)

    # some cell and PD constants
    pd_rate = 0.01
    beta = .01
    prod_upper_lim = 0.6
    prod_lower_lim = 0.1

    # Get results from cells

    Cells = [deepcopy(nodes)] + [deepcopy(diffuse(nodes, dx2, dy2, D, i, pd_rate, beta))
                                 for i in range(1, nt)]

    t2 = time.time()
    print('{0} took {1:.2f} seconds  to simulate'.format(nt, t2-t1))

    print('Sum of all cells: {0}'.format(nodes_to_array(Cells[-1]).sum()))

    # fig, ax = plt.subplots(1, figsize=(10, 10))

    # def animate(i):
    #     ax.cla()
    #     draw_as_network(Cells[i], ax, title='Time elapsed: {0}:{1}:{2}'.format(
    #         i//(60*60), (i//60) % 60, i % 60))

    # anim = animation.FuncAnimation(
    #     fig, animate, frames=range(0, nt-1, 60), blit=False, interval=200)

    # anim.save('/Users/hughesn/nodes-pd_really_low.mp4')

    # print('{0} took {1:.2f} seconds  to animate'.format(nt, time.time()-t2))
