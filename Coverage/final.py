from __future__ import print_function, division

from numpy import *
import networkx as nx
from networkx import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from heapq import heappop, heappush
from itertools import count
from random import randint





def cordgen(N):
    cords_set = set()
    while len(cords_set) < (N * 2):
        x, y = 0, 0
        while (x, y) == (0, 0):
            x, y = randint(0, N - 1), randint(0, N - 1)
        # that will make sure we don't add (7, 0) to cords_set
        cords_set.add((x, y))
    # create discrete colormap
    cmap = colors.ListedColormap(['black', 'white'])
    bounds = [0, N, N * 2]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    coordinates = np.floor(np.random.random((N, N))-0.5)
    return coordinates


def gridplot(coordinates):
    plt.subplots()
    plt.imshow(coordinates, cmap='Greys', interpolation='nearest')
    plt.show()


def mst(cords):
    G=nx.grid_2d_graph(N,N)
    G.remove_nodes_from(cords)
    print(G)
    pos = dict( (n, n) for n in G.nodes() )
    labels = dict( ((i, j), i + (N-1-j) * N ) for i, j in G.nodes() )
    nx.relabel_nodes(G,labels,False)
    inds=labels.keys()
    vals=labels.values()
    inds=[(N-j-1,N-i-1) for i,j in inds]
    pos2=dict(zip(vals, inds))
    T=nx.minimum_spanning_tree(G, algorithm='prim') #kruskal #boruvka
    #   plt.figure(dpi=96, figsize=(N/96,N/96))
    nx.draw_networkx(T, pos=pos2, with_labels=False, node_size = 25)
    plt.show()


if __name__ == '__main__':
    N = 5
    points = cordgen(N)
    gridplot(points)
    print(points)

