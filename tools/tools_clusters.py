# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:56:20 2024

@author: Lucía Lopez
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


def DBCV_DBSCAN(X, eps_range, min_samples_range):
    
    param_grid = {
        'eps': eps_range, 
        'min_samples': min_samples_range
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
                best_per_cluster_validity_index = np.round(per_cluster_validity_index, decimals = 2)
        except:
            continue
    
    return best_per_cluster_validity_index, best_index, best_params, best_labels
        

    

        

M = 30  # Number of clusters
N = 30  # Minimum points in each cluster
E = 0.01  # Radius
xlim = (0, 5)  # X-axis limits
ylim = (0, 5)  # Y-axis limits
min_cluster_distance = 3  # Minimum distance between cluster centers
num_noise_points = 100 # Number of noise points
eps_range = np.arange(0.1,0.5,0.05)
min_samples_range = np.arange(15,25,1)


clusters, noise_points = generate_clusters(M, N, E, xlim, ylim, min_cluster_distance, num_noise_points, variable_radius=True)
plot_clusters(clusters, noise_points)

all_points = np.concatenate(clusters + [noise_points])  # Include noise points
xdata = all_points[:,0]
ydata = all_points[:,1]

[best_per_cluster_validity_index,best_ind, best_param, best_labels] = DBCV_DBSCAN(all_points, eps_range, min_samples_range)

best_eps = np.round(best_param['eps'], decimals = 2)
best_min_samples = np.round(best_param['min_samples'], decimals = 2)


X = all_points

for label in set(best_labels):
    
    if label == -1:
        noise = X[best_labels == label]
        plt.scatter(noise[:, 0], noise[:, 1], c='black', marker='x', label='Noise')
    else:
        cluster = X[best_labels == label]
        cm_list = [] 
        for i in range(np.max(best_labels)+1):
            idx = np.where(best_labels==i)
            x_i = X[:, 0][idx]
            y_i = X[:, 1][idx]
            cm_list.append(np.array([np.mean(x_i),np.mean(y_i)]))
        cmlist = np.array(cm_list)
        plt.scatter(cluster[:, 0], cluster[:, 1])
        plt.scatter(cmlist[:,0], cmlist[:,1], c = 'k', marker = "+", alpha = 1)

        for i, txt in enumerate(best_per_cluster_validity_index):
            plt.annotate(best_per_cluster_validity_index[i], cmlist[i])    