# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:33:19 2023

@author: Lucia
"""

import numpy as np
import h5py as h5
import os
from matplotlib import cm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import hdbscan



os.chdir(r'\\Fileserver\na\Alan Szalai\MPS analysis - septiembre 2023\data ejemplo')

# Define filename
filename = "ROI6_spectrin_locs_drift_corrected_apicked_3.hdf5"

# Read H5 file
f = h5.File(filename, "r")
dataset = f['locs']

# Load  input HDF5 file
frame = dataset['frame']
photon_raw = dataset['photons']
bg = dataset['bg']          

xdata = dataset['x'] 
ydata = dataset['y'] 
zdata = dataset['z'] 

# define px size in nm
pxsize = 133

# Convert x,y values from 'camera subpixels' to nanometres
xdata = xdata * pxsize
ydata = ydata * pxsize

X = np.column_stack((xdata,ydata,zdata))


#Determine the optimal value of epsilon for the DBSCAN

   
# neighbors = NearestNeighbors(n_neighbors=min_samples)
# neighbors_fit = neighbors.fit(X)
# distances, _ = neighbors_fit.kneighbors(X)
# k_distances = np.sort(distances[:, -1])
# epsest = k_distances[-1]

db = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True).fit(X)
dblabels = db.labels_


# db = DBSCAN(eps=50, min_samples=20).fit(X)
# dblabels = db.labels_

cm_list = [] 
   
for i in range(np.max(dblabels)):
    idx = np.where(dblabels==i)
    x_i = xdata[idx]
    y_i = ydata[idx]
    z_i = zdata[idx]
    cm_list.append(np.array([np.mean(x_i),np.mean(y_i),np.mean(z_i)]))




# Remove the noise
range_max = len(X)
Xc = np.array([X[i] for i in range(0, range_max) if dblabels[i] != -1])
labels = np.array([dblabels[i] for i in range(0, range_max) if dblabels[i] != -1])

cmapz = cm.get_cmap('viridis', np.size(labels))
col = cmapz.colors
col = np.delete(col, np.s_[3], axis=1)
col = 255*col

# Generate scatter plot 
plt.figure()
plt.scatter(X[:,0], X[:,1], c='gray', s=1, alpha = 0.5)
plt.figure()
plt.scatter(Xc[:,0], Xc[:,1], c = labels, marker="+", alpha = 0.5)


cm = np.array(cm_list)
    
X = 3 #study distances until Xth neighbor, 
tree = KDTree(cm)
distances, indexes = tree.query(cm, X+1) 
distances = distances[:,1:] # exclude distance to the same molecule; distances has N rows (#clusters) and M columns (# neighbors)
indexes = indexes[:,1:]   

distances = np.ndarray.flatten(distances)

plt.figure()
plt.hist(distances, bins=50)


# histd, bin_edgesd = np.histogram(distances, bins=40)
# widthd = np.mean(np.diff(bin_edgesd))
# bincentersd = np.mean(np.vstack([bin_edgesd[0:-1],bin_edgesd[1:]]), axis=0)

# fig, ax = plt.subplots()

# # the histogram of the data
# ax.hist(distances, bins=40)

# plt.hist(distances)


   