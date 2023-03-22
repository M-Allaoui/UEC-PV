import pandas as pd
d = pd.read_csv("C:/Users/GOOD DAY/Desktop/Classeur1.csv")

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
#if using a Jupyter notebook, include:

X = np.array([0, 0.2, 0.8,1])
Y = np.array([0, 0.2, 0.8,1])
X,Y = np.meshgrid(X,Y)
#data=np.array([[0.949, 0.951, 0.957,	0.958],[0.953,	0.955,	0.96,	0.951],[0.965,	0.966,	0.97,	0.971],[0.971,	0.971,	0.972,	0.972]])
data=np.array([[0.94,	0.942,	0.948,	0.949],[0.948,	0.95,	0.951,	0.951],[0.964,	0.965,	0.969,	0.97],[0.97,	0.97,	0.971,	0.971]])

#Z = X*np.exp(-X - Y)


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

mycmap = plt.get_cmap('gist_earth')
#ax.set_title('Studding the effect of alpha and beta on the perfomance of our algorithm using accuracy measure.')
# Plot a 3D surface
surf1=ax.plot_surface(X, Y, data, cmap=mycmap)
fig.colorbar(surf1, ax=ax)
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_zlabel('Accuracy')

plt.show()

