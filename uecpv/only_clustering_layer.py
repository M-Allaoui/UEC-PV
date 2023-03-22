import pandas as pd
import numpy as np
import numba
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from sklearn.utils.linear_assignment_ import linear_assignment

def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w


def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

@numba.njit("f4(f4[:],f4[:])", fastmath=True)
def rdist(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2

    return result

@numba.njit(fastmath=True, parallel=True)
def S(embedding, centroids, a, b):

    q=np.zeros((embedding.shape[0],centroids.shape[0]))
    #q=q.astype(np.float64)
    for i in range(embedding.shape[0]):
        data=embedding[i]
        for k in range(centroids.shape[0]):
            dist_squared = rdist(data, centroids[k])
            q[i,k]=pow((1.0 + a * pow(dist_squared, b)),-1.0)
            #q[i, k] = pow(dist_squared, -1.0)
            #q[i, k] = pow((1.0 + dist_squared), -1.0)


    return q

#computethe atrget variable P
@numba.njit(fastmath=True, parallel=True)
def T(q):
    weight = q ** 2 / (q.sum(0))
    #print("sum q= ", q.sum(0))
    #print(weight.sum(1))
    Target=(weight.T / (weight.sum(1))).T
    #print("target=",(weight.T / (weight.sum(1))).T)
    return Target

@numba.njit(fastmath=True, nopython=False)
def clustering_layout_SGD(embedding, centroids, initial_alpha, a, b, n_epochs, y_pred):
    #initial_alpha=1e-30
    alpha=initial_alpha
    update_interval=20
    tol=1e-6
    y_pred_last=y_pred
    n_clusters=centroids.shape[0]
    for n in range(n_epochs):
        if n % update_interval == 0:
            soft = S(embedding, centroids, a, b)
            target = T(soft)
        for i in range(embedding.shape[0]):
            for k in range(n_clusters):

                centroid=centroids[k]
                dist_squared = rdist(embedding[i], centroid)
                #dist_squared = rdist2(embedding[i], centroid)
                if dist_squared > 0.0:
                    grad_coeff = 2.0 * a * b * pow(dist_squared, b - 1.0) * target[i, k]
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                    #grad_coeff = dist_squared * target[i, k]
                    #grad_coeff /= pow(dist_squared, 3)
                else:
                    grad_coeff = 0.0

                #grad_coeff = clip(grad_coeff)
                #grad_coeff = grad_coeff +0.5
                embedding[i] += grad_coeff * alpha
        for k in range(n_clusters):
          centroid = centroids[k]
          for i in range(embedding.shape[0]):
              x = embedding[i]
              dist_squared = rdist(x, centroid)
              #dist_squared = rdist2(x, centroid)
              if dist_squared > 0.0:
                  grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0) * target[i, k]
                  grad_coeff /= a * pow(dist_squared, b) + 1.0
                  #grad_coeff = dist_squared * target[i, k]
                  #grad_coeff /= pow(dist_squared, 3)
              else:
                  grad_coeff = 0.0
              #grad_coeff = clip(grad_coeff)
              #grad_coeff = grad_coeff + 0.5
              centroid += grad_coeff * alpha
        alpha = initial_alpha * (
            1.0 - (float(n) / float(n_epochs))
        )
        if n % int(n_epochs /5) == 0:
            print(
                "\tcompleted ", n, " / ", n_epochs, "epochs", "Lr= ", alpha
            )
    return embedding, centroids

d = pd.read_csv("E:/PycharmProjects/Datasets and AE whieghts/datasets/USPS/usps_ae_CAE.csv")
y = pd.read_csv("E:/PycharmProjects/Datasets and AE whieghts/datasets/USPS/USPS_y.csv")

d=np.array(d)
d=d.astype(np.float32)
print(np.shape(d))
y=np.array(y)
y = y.reshape((y.shape[0]))
print(np.shape(y))

a=  1.9328083975445773
b=  0.7904949732966567

kmeans = cluster.KMeans(n_clusters=10, random_state=0)
y_pred = kmeans.fit_predict(d)
centroids = kmeans.cluster_centers_
centroids=centroids.astype(np.float32)

acc = np.round(cluster_acc(y, y_pred),5)
print("Accuracy kmeans labels: ",acc)

print(d.dtype)
print(centroids.dtype)
embedding, centroids = clustering_layout_SGD(d, centroids, 0.01, a, b, 1000, y_pred)
y_pred = S(embedding, centroids, a, b)
y_pred = y_pred.argmax(1)
acc = np.round(cluster_acc(y, y_pred),5)
print("Accuracy clustering layer labels: ",acc)

plt.scatter(embedding[:, 0], embedding[:, 1], c=y, s=1, cmap='Spectral')
plt.scatter(centroids[:,0], centroids[:, 1], s=3)
plt.show();