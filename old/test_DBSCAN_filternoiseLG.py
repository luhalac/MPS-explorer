# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:29:35 2023

@author: Lucia
"""



import numpy as np
import h5py as h5
import os
from matplotlib import cm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from scipy import interpolate


def filter_noise(X, K):
   
    Atot = np.column_stack((X[:,0],X[:,1]))
    A_clus = Atot
   
    idx_noise = []
   
    Lx = np.max(A_clus[:, 0]) - np.min(A_clus[:, 0])
    Ly = np.max(A_clus[:, 1]) - np.min(A_clus[:, 1])
    D = len(A_clus) / (Lx * Ly)
   
    while True:
        tree = KDTree(A_clus)
        distances, indexes = tree.query(A_clus, K + 1)
        distances = distances[:, 1:]
        indexes = indexes[:, 1:]
       
        r = np.sort(distances[:, -1])
        maxr, minr = np.max(r), np.min(r)
        r_ev = np.arange(minr, maxr)
       
        cumulative = np.arange(0, len(r)) / len(r)
        λ = D * np.pi
       
        PDF_U2Dn_r = np.exp(-λ * r_ev ** 2) * r_ev ** (2 * K - 1)
        PDF_U2Dn_c = (2 / np.math.factorial(K - 1)) * (λ ** K)
        PDF_U2Dn = PDF_U2Dn_r * PDF_U2Dn_c
       
        CDF_U2Dn = np.cumsum(PDF_U2Dn)
       
        f1 = interpolate.interp1d(r_ev, CDF_U2Dn, kind='linear')
        f2 = interpolate.interp1d(r, cumulative, kind='linear')
       
        xx = np.linspace(max(r_ev[0], r[0]), min(r_ev[-1], r[-1]), 1000)
        y1_interp = f1(xx)
        y2_interp = f2(xx)
       
        idx = np.argwhere(np.diff(np.sign(y1_interp - y2_interp))).flatten()
        delta = 50
        idx = idx[(idx > (minr + delta))]
       
        if len(idx) == 0:
            break
       
        idx_min = idx[idx > 0]
        mindist = xx[idx_min]
       
        if len(mindist) > 0:
            idx_noise_i = np.where(distances[:, -1] > mindist)[0]
            idx_clustered = np.where(distances[:, -1] < mindist)[0]
           
            A_clus = A_clus[idx_clustered]
            idx_noise.append(idx_noise_i)
        else:
            break
   
    flat_list = np.concatenate(idx_noise)
    noise = Atot[flat_list]
    clus = A_clus
   
   
    return clus, noise


os.chdir(r'\\Fileserver\na\Alan Szalai\MPS analysis - septiembre 2023\data ejemplo')

# Define filename
filename = "ROI6_spectrin_locs_drift_corrected_apicked_1.hdf5"

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


clus,noise = filter_noise(X,10)

#Determine the optimal value of epsilon for the DBSCAN

   
# neighbors = NearestNeighbors(n_neighbors=min_samples)
# neighbors_fit = neighbors.fit(X)
# distances, _ = neighbors_fit.kneighbors(X)
# k_distances = np.sort(distances[:, -1])
# epsest = k_distances[-1]


db = DBSCAN(eps=50, min_samples=20).fit(clus)
dblabels = db.labels_

cm_list = [] 
   
for i in range(np.max(dblabels)):
    idx = np.where(dblabels==i)
    x_i = xdata[idx]
    y_i = ydata[idx]
    z_i = zdata[idx]
    cm_list.append(np.array([np.mean(x_i),np.mean(y_i),np.mean(z_i)]))




# Remove the noise
# range_max = len(X)
# Xc = np.array([clus[i] for i in range(0, range_max) if dblabels[i] != -1])
# labels = np.array([dblabels[i] for i in range(0, range_max) if dblabels[i] != -1])

# cmapz = cm.get_cmap('viridis', np.size(labels))
# col = cmapz.colors
# col = np.delete(col, np.s_[3], axis=1)
# col = 255*col

# Generate scatter plot 
plt.figure()
plt.scatter(X[:,0], X[:,1], c='gray', s=1, alpha = 0.5)
plt.scatter(clus[:,0], clus[:,1], c = 'black', marker="+", alpha = 0.5)


# cm = np.array(cm_list)
    
# X = 10 #study distances until Xth neighbor, 
# tree = KDTree(cm)
# distances, indexes = tree.query(cm, X+1) 
# distances = distances[:,1:] # exclude distance to the same molecule; distances has N rows (#clusters) and M columns (# neighbors)
# indexes = indexes[:,1:]   

# distances = np.ndarray.flatten(distances)

# plt.figure()
# plt.hist(distances, bins=50)


# histd, bin_edgesd = np.histogram(distances, bins=40)
# widthd = np.mean(np.diff(bin_edgesd))
# bincentersd = np.mean(np.vstack([bin_edgesd[0:-1],bin_edgesd[1:]]), axis=0)

# fig, ax = plt.subplots()

# # the histogram of the data
# ax.hist(distances, bins=40)

# plt.hist(distances)