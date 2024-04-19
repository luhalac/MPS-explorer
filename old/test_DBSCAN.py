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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from hdbscan.validity import validity_index


def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))




def generate_clusters(M, N, E, xlim, ylim, min_cluster_distance, num_noise_points=0, variable_radius=True):
    clusters = []
    
    if variable_radius:
        radius_range = np.arange(E - 0.1, E + 0.2, 0.05)
    else:
        radius_range = [E]
    
    for _ in range(M):
        cluster = []
        current_radius = np.random.choice(radius_range)
        # Generate a random center for the cluster
        while True:
            center = np.random.uniform(low=xlim[0], high=xlim[1]), np.random.uniform(low=ylim[0], high=ylim[1])
            if all(distance(np.array(center), np.array(existing_center)) > min_cluster_distance + current_radius for existing_center in clusters):
                break
        while len(cluster) < N:
            # Generate points within the neighborhood of the cluster center
            point = np.array(center) + np.random.uniform(-current_radius, current_radius, size=2)
            if xlim[0] <= point[0] <= xlim[1] and ylim[0] <= point[1] <= ylim[1]:
                cluster.append(point)
        clusters.append(cluster)
    
    # Generate noise points
    noise_points = []
    for _ in range(num_noise_points):
        noise_point = np.random.uniform(low=xlim[0], high=xlim[1]), np.random.uniform(low=ylim[0], high=ylim[1])
        noise_points.append(noise_point)
    
    return clusters, noise_points


def plot_clusters(clusters, noise_points):
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')
    if noise_points:
        noise_points = np.array(noise_points)
        plt.scatter(noise_points[:, 0], noise_points[:, 1], c='black', marker='x', label='Noise Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clusters vs noise')

    plt.subplot(1, 2, 2)
    all_points = np.concatenate(clusters)
    all_points = np.concatenate([all_points, noise_points])
    plt.scatter(all_points[:, 0], all_points[:, 1], c='black', marker='.')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Indistinguishable points')

    plt.tight_layout()
    plt.show()


def DBCV_DBSCAN(X):
    param_grid = {
        'eps': np.arange(0.1,0.5,0.05),
        'min_samples': np.arange(15,20,5)
    }
   
    best_index = -1
    best_params = None
    best_labels = None  
   
    for params in ParameterGrid(param_grid):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        db = DBSCAN(**params).fit(X)
        labels = db.labels_
       
        try:
            [index,per_cluster_validity_index] = validity_index(X_scaled, labels, per_cluster_scores=True)
            if index > best_index:
                
                best_index = index
                best_params = params
                best_labels = labels
        except:
            continue

    return per_cluster_validity_index, best_index, best_params, best_labels



#%% Random clusters generation


M = 20  # Number of clusters
N = 20  # Minimum points in each cluster
E = 0.2  # Radius
xlim = (0, 5)  # X-axis limits
ylim = (0, 5)  # Y-axis limits
min_cluster_distance = 3  # Minimum distance between cluster centers
num_noise_points = 50 # Number of noise points


clusters, noise_points = generate_clusters(M, N, E, xlim, ylim, min_cluster_distance, num_noise_points, variable_radius=True)
plot_clusters(clusters, noise_points)

all_points = np.concatenate(clusters + [noise_points])  # Include noise points


[per_cluster_validity_index,best_ind, best_param, best_labels] = DBCV_DBSCAN(all_points)


# os.chdir(r'C:/Users/Usuario/Desktop/axones filtrados/230914 subROIs (Axons)/ROI2')

# # Define filename
# filename = "ROI2_B2spectrin_unified_locs_rcc_apicked_1_filter.hdf5"

# # Read H5 file
# f = h5.File(filename, "r")
# dataset = f['locs']

# # Load  input HDF5 file
# frame = dataset['frame']
# photon_raw = dataset['photons']
# bg = dataset['bg']          

# xdata = dataset['x'] 
# ydata = dataset['y'] 
# zdata = dataset['z'] 

# # define px size in nm
# pxsize = 133

# # Convert x,y values from 'camera subpixels' to nanometres
# xdata = xdata * pxsize
# ydata = ydata * pxsize

# X = np.column_stack((xdata,ydata))

# # Convert the data to double precision floating-point
# X = X.astype(np.double)

#Determine the optimal value of epsilon for the DBSCAN




# db = hdbscan.HDBSCAN(min_cluster_size=10).fit(X)
# dblabels = db.labels_


# # Define the ranges for eps and min_samples
# eps_range = np.arange(15, 50, 5)  # Adjusted range to include upper limit
# min_samples_range = np.arange(10, 30, 5)

# # Initialize variables to store the best parameters and corresponding validity index
# best_eps = None
# best_min_samples = None
# best_validity_index = -float('inf')  # Initialize with negative infinity

# # Iterate over all combinations of eps and min_samples
# for eps in eps_range:
#     for min_samples in min_samples_range:
#         # Perform DBSCAN clustering
#         db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
#         db_labels = db.labels_  # Assuming X is your data
        
#         # Calculate validity index
#         validity_index = hdbscan.validity.validity_index(X, db_labels)
        
#         # Update best parameters if validity index is higher
#         if validity_index > best_validity_index:
#             best_eps = eps
#             best_min_samples = min_samples
#             best_validity_index = validity_index

# # Print the best parameters and validity index
# print("Best eps:", best_eps)
# print("Best min_samples:", best_min_samples)
# print("Best validity index:", best_validity_index)



# db = DBSCAN(eps=best_eps, min_samples=best_min_samples).fit(X)
# dblabels = db.labels_



dblabels = best_labels


# validity = hdbscan.validity_index(X, dblabels)

cm_list = [] 

   
for i in range(np.max(dblabels)+1):
    idx = np.where(dblabels==i)
    x_i = xdata[idx]
    y_i = ydata[idx]

    cm_list.append(np.array([np.mean(x_i),np.mean(y_i)]))

cmlist = np.array(cm_list)


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
fig, ax = plt.subplots()
ax.scatter(Xc[:,0], Xc[:,1], c = labels, marker="+", alpha = 0.5)
ax.scatter(cmlist[:,0], cmlist[:,1], c = 'k', marker = "+", alpha = 1)

for i, txt in enumerate(per_cluster_validity_index):
    ax.annotate(per_cluster_validity_index[i], cmlist[i])


# cm = np.array(cm_list)
    
# Kn = 3 #study distances until Xth neighbor, 
# tree = KDTree(cm)
# distances, indexes = tree.query(cm, Kn+1) 
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

   