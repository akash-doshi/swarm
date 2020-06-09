from __future__ import print_function, division

from numpy import *
import time
import networkx as nx
from networkx import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from heapq import heappop, heappush
from itertools import count
from random import randint
from PIL import Image

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
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


def cordgen(M, N, seed):
    np.random.seed(seed)
    coordinates = np.floor(np.random.random((M, N))-0.5)
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
    nx.draw_networkx(T, pos=pos2, with_labels=False, node_size = 10)
    plt.show()


def setkm(ncluster):

    km = KMeans(
        n_clusters=ncluster, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    return km


def setAG(ncluster):
    km = AgglomerativeClustering(
        n_clusters=ncluster)
    return km

def setDB(ncluster):
    km = DBSCAN(
        eps=1.43)
    return km

def setSC(ncluster):
    km = SpectralClustering(
        n_clusters=ncluster,
        n_init=40,
        affinity='rbf'
    )
    return km


def genAG(ag, x, n):
    y_km = ag.fit_predict(x)
    coords = []

    for i in range(n):
        plt.scatter(
            x[y_km == i, 0], x[y_km == i, 1],
            s=50, marker='o', edgecolor='black',
            label='cluster ' + str(i+1)
        )
        #print((x[y_km == i, 0]).tolist())
        result = (zip((x[y_km == i, 0]).tolist(), (x[y_km == i, 1])))
        leest = list(result)
        coords.append(leest)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig('example1.png', dpi=300, format='png')
    plt.show()
    print(coords)
    return coords


def genkm(km, x, n):
    y_km = km.fit_predict(x)
    coords = []

    for i in range(n):
        plt.scatter(
            x[y_km == i, 0], x[y_km == i, 1],
            s=50, marker='o', edgecolor='black',
            label='cluster ' + str(i+1)
        )
        #print((x[y_km == i, 0]).tolist())
        result = (zip((x[y_km == i, 0]).tolist(), (x[y_km == i, 1])))
        leest = list(result)
        coords.append(leest)
    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='.',
        c='black', edgecolor='black',
        label='centroids'
    )
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig('example1.png', dpi=300, format='png')
    plt.show()
    print(coords)
    return coords


def gen_mst(cords):
    start = time.perf_counter()
    G=nx.grid_2d_graph(N,N)
    #G.clear()
    G.remove_nodes_from(cords)
    #newnodes = nx.Graph(cords)
    #G.add_node(newnodes)
    labels = dict( ((i, j), i + (N-1-j) * N ) for i, j in G.nodes() )
    nx.relabel_nodes(G,labels,False)
    inds=labels.keys()
    vals=labels.values()
    inds=[(N-j-1,N-i-1) for i,j in inds]
    pos2=dict(zip(vals, inds))
    flipped_pos = {node: (y, x) for (node, (x, y)) in pos2.items()}
    T=nx.minimum_spanning_tree(G, algorithm='prim') #kruskal #boruvka
    end = time.perf_counter()

    perim = 0
    turn = 0
    unconnected = 0
    for node in list(T.nodes):

        nodeedge = edges(T, node)
        adjdict = T.__getitem__(node)
        #print(nodeedge)
        turncount = 0
        perimcount = 2

        if not nodeedge:
            # This is a not connected to anything Type Z
            turncount = 0
            perimcount = 0
            unconnected += 1
        elif len(nodeedge) == 1:
            #this is an end piece Type A
            turncount = 2
        elif len(nodeedge) == 2:
            #could be straight line or corner (B or C)
            adjlist = []
            for adj in adjdict:
                adjlist.append(list(pos2[adj]))
            if adjlist[0][0] == adjlist [1][0] or adjlist[0][1] == adjlist [1][1]:
                #Type C
                turncount = 0
            else:
                #Type C
                turncount = 2
        elif len(nodeedge) == 3:
            #T shap e
            turncount = 2
        elif len(nodeedge) == 4:
            #Cross shape
            turncount = 4
        perim += perimcount #len(nodeedge)
        turn += turncount
    print(f"\nMST generated in {end - start:0.6f} seconds:")

    print("\ntotal edge perimeter: " + str(perim))
    print("total turns: " + str(turn))
    print("total nodes: " + str(len(T.nodes)))
    print("unconnected nodes: " + str(unconnected))


    plt.figure(figsize=(8,8), dpi=768)
    fig, ax = plt.subplots()
    nx.draw_networkx(T, pos=flipped_pos, with_labels=False, node_size = 25, ax=ax)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlim([-0.5, N-0.5])
    ax.set_ylim([-0.5, N-0.5])
    #plt.gca().invert_yaxis()
    #plt.gca().invert_xaxis()

    plt.show()


def loadmap(m):
    img = Image.open(m).convert('L')
    mEdit = np.array(img)
    mEdit = ~mEdit
    mEdit[mEdit > 0] = 1
    gridplot(mEdit)
    print(mEdit)
    return mEdit


if __name__ == '__main__':

    random = False
    seed = 12
    clusterSize = 8


    if random:
        M = 8
        N = 8
        points = cordgen(M, N, seed)
        gridplot(points)
        points += 1
    else:
        mapname = 'map1.png'
        points = loadmap(mapname)
        M, N = points.shape

    grid = [tuple(brug) for brug in np.argwhere((np.zeros((M,N))) == 0).tolist()]
    coords = [tuple(brug) for brug in np.argwhere(points == -1).tolist()]
    data = (np.flip(np.argwhere(points == 0)))
    km = setSC(clusterSize)
    plotpoints = genAG(km, data, clusterSize)
    #for item in plotpoints:
    #    throwlist = [brug for brug in grid if brug not in item]
    #    gen_mst(throwlist)


