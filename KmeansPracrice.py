from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data, target = load_iris(return_X_y = True)

#print (data, target)
SSE = {}
for i in range(1, 10):
	kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
	SSE[i] = kmeans.inertia_
print (SSE)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(list(SSE.keys()), list(SSE.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
ax.annotate('elbow', xy=(2, 150), xytext=(3, 170), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('elbow', xy=(3, 90), xytext=(3, 170), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
fig.savefig('KmeansElbow.jpeg')

