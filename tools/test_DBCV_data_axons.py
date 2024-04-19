# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:24:21 2024

@author: Usuario
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



# load exp data
os.chdir(r'C:\Users\Usuario\Desktop\axones filtrados\230914 subROIs (Axons)\ROI3')

# Define filename
filename = "ROI3_B2spectrin_unified_locs_filter_rcc_apicked_2_filter.hdf5"


# Read H5 file
f = h5.File(filename, "r")
dataset = f['locs']

# define px size in nm
pxsize = 133
# # Load  input HDF5 file
frame = dataset['frame']
photon_raw = dataset['photons']
bg = dataset['bg']     
xdata = dataset['x'] 
ydata = dataset['y']      

# Convert x,y values from 'camera subpixels' to nanometres
xdata = xdata * pxsize
ydata = ydata * pxsize

X = np.column_stack((xdata,ydata))

# Convert the data to double precision floating-point
X = X.astype(np.double)

# select params range to explore
eps_range = np.arange(20,40,5)
min_samples_range = np.arange(10,20,5)


[best_per_cluster_validity_index,best_ind, best_param, best_labels] = DBCV_DBSCAN(X, eps_range, min_samples_range)

best_eps = np.round(best_param['eps'], decimals = 2)
best_min_samples = np.round(best_param['min_samples'], decimals = 2)


# Generate scatter plot 

for label in set(best_labels):
    
    if label == -1:
        noise = X[best_labels == label]
        plt.scatter(noise[:, 0], noise[:, 1], c='gray', s=1, alpha = 0.5, label='Noise')
    else:
        cluster = X[best_labels == label]
        cm_list = [] 
        for i in range(np.max(best_labels)+1):
            idx = np.where(best_labels==i)
            x_i = X[:, 0][idx]
            y_i = X[:, 1][idx]
            cm_list.append(np.array([np.mean(x_i),np.mean(y_i)]))
        cmlist = np.array(cm_list)
        plt.scatter(cluster[:, 0], cluster[:, 1], marker="+", alpha = 0.5)
        plt.scatter(cmlist[:,0], cmlist[:,1], c = 'k', marker = "+", alpha = 1)

        for i, txt in enumerate(best_per_cluster_validity_index):
            plt.annotate(best_per_cluster_validity_index[i], cmlist[i])  