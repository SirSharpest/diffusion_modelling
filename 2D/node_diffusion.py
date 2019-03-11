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


def D_eff(D, q, cell_um):
    return (D*q*cell_um)/(D+q*cell_um)


def do_node_diffusion(nodes, dx2, D, ts, pd_rate, b):
    """
    nodes a np array of nodes
    dx2, dx2 the difference in x,y squared
    D diffusion constant
    f a method to make a diffusion matrix of available cells
    """
    u = nodes_to_array(nodes).copy()
    array_update_nodes(diffuse(u, dx2, D, pd_rate, b), nodes)
    return nodes


def diffuse(u, dx2, D, pd_rate, b, dt):
    un = u.copy()
    Y, X = un.shape
    # TODO: conn = connectivity(un, pd_rate)
    # TODO: beta = production(un, b)
    for y in range(0, Y):
        for x in range(0, X):
            # Flux in the diffusion equation is positional dependant
            # therefore, the order of the variables is of the uptmost importance
            # Moving from left to right, up to down this becomes:
            Cs = []
            if y > 0:
                # Check if upper is available
                Cs.append(u[y-1, x])
            if y < Y-1:
                # Check if lower is available
                Cs.append(u[y+1, x])
            if x > 0:
                # Check if left is available
                Cs.append(u[y, x-1])
            if x < X-1:
                # Check if right is available
                Cs.append(u[y, x+1])
            if len(Cs) < 2:
                un[y, x] = diffuse_1point(u[y, x], D, *Cs, dx2, b, dt)
            elif len(Cs) < 3:
                un[y, x] = diffuse_2point(u[y, x], D, *Cs, dx2, b, dt)
            elif len(Cs) < 4:
                un[y, x] = diffuse_3point(u[y, x], D, *Cs, dx2, b, dt)
            elif len(Cs) == 4:
                un[y, x] = diffuse_4point(u[y, x], D, *Cs, dx2, b, dt)
            else:
                # No connectivity
                continue
    return un


def production(c, b):
    return np.zeros(c.shape)  # + (b * np.where)


def diffuse_1point(c0, D, x1, dx2, b, dt):
    c = c0 + dt*D * ((x1 - c0)/dx2) + b
    return c


def diffuse_2point(c0, D, x1, x2, dx2, b, dt):
    c = c0 + dt*D * ((x1 - 2*c0 + x2)/dx2) + b
    return c


def diffuse_3point(c0, D, x1, x2, y1, dx2, b, dt):
    c = c0 + dt*D * ((x1 - 3*c0 + x2 + y1)/dx2) + b
    return c


def diffuse_4point(c0, D, x1, x2, y1, y2, dx2, b, dt):
    c = c0 + dt*D * ((x1 - 4*c0 + x2 + y1 + y2)/dx2) + b
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


def array_normalise(arr):
    """
    What percentage of the total sum is each node

    This also assumes no loss...
    """

    return arr/arr.sum()


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
    arr = array_normalise(nodes_to_array(nodes))
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

            labels[cur_node] = "~{0:.2f}".format(
                np.around(lbl, 4)*100) if lbl > cut_off_of_interest else ''

    with np.errstate(divide='ignore'):
        sizes = sizes.ravel()*10000

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
