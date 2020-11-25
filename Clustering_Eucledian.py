
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
X=np.array([[29,0],[33,0],[35,0],[37,0],[41,0],[43,0],[47,0],[51,0],[53,0],[60,0],[64,0],[70,0]])
labelslist= range(1,13)
fig=plt.figure(figsize=(15,10))
fig.suptitle('Scatter Plot of Data Points')
plt.scatter(X[:,0],X[:,1],label='True Position')
linked= linkage(X,'average',metric='euclidean')
fig=plt.figure(figsize=(15,10))
fig.suptitle('Clustering Dendrogram in Average Linkage(Centroid) Case')
dendrogram(linked,orientation='top',labels=labelslist,distance_sort='descending',show_leaf_counts=True)
cluster= AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='average')
y=cluster.fit_predict(X)
print("Clustering in Average Linkage (Centroid) Case")
print(y)

""" Clustering in Single Linkage( Minimimum of any Two)"""
linked= linkage(X,'single',metric='euclidean')
fig=plt.figure(figsize=(15,10))
fig.suptitle('Clustering Dendrogram in Single Linkage(Minimum of any Two) Case')
dendrogram(linked,orientation='top',labels=labelslist,distance_sort='descending',show_leaf_counts=True)
cluster1= AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='single')
z=cluster1.fit_predict(X)
print("Clustering in Single Linkage (Minimum of any Two) Case")
print(z)
plt.show()

