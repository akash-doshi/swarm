from __future__ import print_function, division

from numpy import *
from time import time
import networkx as nx
from networkx import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from heapq import heappop, heappush
from itertools import count
from random import randint

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def oldgen(N):
    cords_set = set()
    while len(cords_set) < (N * 2):
        x, y = 0, 0
        while (x, y) == (0, 0):
            x, y = randint(0, N - 1), randint(0, N - 1)
        # that will make sure we don't add (7, 0) to cords_set
        cords_set.add((x, y))
    # create discrete colormap
    print("Coords:")
    print(cords_set)
    cmap = colors.ListedColormap(['black', 'white'])
    bounds = [0, N, N * 2]
    norm = colors.BoundaryNorm(bounds, cmap.N)


def cordgen(N):
    coordinates = np.floor(np.random.random((N, N))-0.5)
    return coordinates


def gridplot(coordinates):
    plt.subplots()
    plt.imshow(coordinates, cmap='Greys', interpolation='nearest')
    plt.gca().invert_xaxis()
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
    #print(number_of_edges(T))
    #print(number_of_nodes(T))
    #print(list(T.nodes))

    perim = 0
    turn = 0
    for node in list(T.nodes):

        nodeedge = edges(T, node)
        adjdict = T.__getitem__(node)
        #print(nodeedge)
        turncount = 0

        if not nodeedge:
            # This is a not connected to anything
            turncount = 0
        elif len(nodeedge) == 1:
            #this is a corner piece
            turncount = 2
        elif len(nodeedge) == 2:
            #could be straight line or corner
            adjlist = []
            for adj in adjdict:
                adjlist.append(list(pos2[adj]))
            if adjlist[0][0] == adjlist [1][0] or adjlist[0][1] == adjlist [1][1]:
                turncount = 0
            else:
                turncount = 2
        elif len(nodeedge) == 3:
            #T shap e
            turncount = 2
        elif len(nodeedge) == 4:
            #Cross shape
            turncount = 4
        perim += len(nodeedge)
        turn += turncount

    print("total edge perimeter: " + str(perim))
    print("total turns: " + str(turn))
    nx.draw_networkx(T, pos=pos2, with_labels=False, node_size = 25)
    plt.show()


def setkm(ncluster):

    km = KMeans(
        n_clusters=ncluster, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    return km


def genkm(km, x, n):
    y_km = km.fit_predict(x)

    for i in range(n):
        plt.scatter(
            x[y_km == i, 0], x[y_km == i, 1],
            s=50, marker='o', edgecolor='black',
            label='cluster ' + str(i+1)
        )

    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='.',
        c='black', edgecolor='black',
        label='centroids'
    )
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    N = 10
    clusterSize = 3
    points = cordgen(N)
    gridplot(points)
    print(points)
    #coords = np.argwhere(points == -1).tolist()
    coords = [tuple(brug) for brug in np.argwhere(points == -1).tolist()]
    print(coords)
    mst(coords)

    data = (np.flip(np.argwhere(points == 0)))

    print(data)
    km = setkm(clusterSize)
    genkm(km, data, clusterSize)



