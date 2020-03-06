from __future__ import print_function, division
import numpy
from numpy import *
from networkx import *
import matplotlib.pyplot as plt
import networkx as nx
from heapq import heappop, heappush
from itertools import count
from random import randint

N=10
#Obstacle Gen
cords_set = set()
while len(cords_set) < (N*2):
    x, y = 0, 0
    while (x, y) == (0, 0):
        x, y = randint(0, N - 1), randint(0, N - 1)
    # that will make sure we don't add (7, 0) to cords_set
    cords_set.add((x, y))
print(cords_set)

G=nx.grid_2d_graph(N,N)
G.remove_nodes_from(cords_set)
print(G)
pos = dict( (n, n) for n in G.nodes() )
labels = dict( ((i, j), i + (N-1-j) * N ) for i, j in G.nodes() )
nx.relabel_nodes(G,labels,False)
inds=labels.keys()
vals=labels.values()
inds=[(N-j-1,N-i-1) for i,j in inds]
pos2=dict(zip(vals, inds))

nx.draw_networkx(G, pos=pos2, with_labels=False, node_size = 25)
T=nx.minimum_spanning_tree(G, algorithm='prim') #kruskal #boruvka
plt.figure()
nx.draw_networkx(T, pos=pos2, with_labels=False, node_size = 25)
plt.show()