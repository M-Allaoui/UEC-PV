import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


d = pd.read_csv("E:/PycharmProjects/Datasets and AE whieghts/datasets/MNIST/mnist_CAE.csv")
y = pd.read_csv("E:/PycharmProjects/Datasets and AE whieghts/datasets/MNIST/mnist_label.csv")
d=np.array(d)
y=np.array(y)
d = PCA(n_components=2).fit_transform(d)
plt.scatter(d[:, 0], d[:, 1], c=y, s=1, cmap='Spectral')
#plt.scatter(centroids[:,0], centroids[:, 1], s=3)
plt.show();