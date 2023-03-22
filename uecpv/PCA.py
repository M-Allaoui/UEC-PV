from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

d = pd.read_csv("C:/Users/GOOD DAY/PycharmProjects/Datasets and AE whieghts/datasets/CIFAR10/cifar10_vgg16.csv")
d=np.array(d)

pca = PCA(n_components=350).fit(d)
"""plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show();"""
X_pca = pca.transform(d)
np.savetxt('C:/Users/GOOD DAY/PycharmProjects/Datasets and AE whieghts/datasets/cifar10_vgg16_pca.csv', X_pca, delimiter=',', fmt='%f')
