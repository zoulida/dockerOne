__author__ = 'zoulida'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition,manifold

def load_data():
    iris=datasets.load_iris()
    print(iris)
    return iris.data,iris.target

def test_PCA(*data):
    X,Y=data
    pca = decomposition.PCA(n_components=None)
    #pca=decomposition.IncrementalPCA(n_components=None) #超大规模分批加载内存
    pca.fit(X)
    print("explained variance ratio:%s"%str(pca.explained_variance_ratio_))

def plot_PCA(*data):
    X,Y=data
    pca=decomposition.PCA(n_components=2)
    pca.fit(X)
    X_r=pca.transform(X)
    print(X_r)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)
    for label,color in zip(np.unique(Y),colors):
        position=Y==label
        print(position)
        print(X_r[position, 0])
        ax.scatter(X_r[position,0],X_r[position,1],label="target=%d"%label,color=color)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
    plt.show()

X,Y=load_data()
test_PCA(X,Y)
plot_PCA(X,Y)