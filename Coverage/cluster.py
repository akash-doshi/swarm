from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

x_iris, y_iris = datasets.load_iris(return_X_y=True)

print(x_iris)

clusterSize = 3

k_means = KMeans(
            n_clusters=clusterSize, init='random',
            n_init=50, max_iter=300,
            tol=1e-4, random_state=0
        )
k_means.fit(x_iris)

plt.scatter(x_iris[:, 0], x_iris[:, 1], label='True Position', c=k_means.labels_, cmap='rainbow')
plt.show()
print(k_means.labels_)


def ClusterIndicesNumpy(clustNum, labels_array): #numpy
    return np.where(labels_array == clustNum)[0]


def ClusterIndicesComp(clustNum, labels_array): #list comprehension
    return np.array([i for i, x in enumerate(labels_array) if x == clustNum])


print(x_iris[ClusterIndicesNumpy(2, k_means.labels_)])
