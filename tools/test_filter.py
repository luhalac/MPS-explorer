# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:33:50 2023

@author: Lucia
"""

import numpy as np
import h5py as h5
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from matplotlib import cm
from skimage.morphology import square, dilation, disk

os.chdir(r'C:\Users\Lucia\Documents\Lu Lopez\SIMPLER\data Chechu jun23')

# Define filename
filename = "SIMPLER1_100pM_9meratto655_2nM_100ms_647nm_130mW_telescope_12.3mm_1_MMStack_Default.ome_locs.hdf5"

# Read H5 file
f = h5.File(filename, "r")
dataset = f['locs']

# Load  input HDF5 file
frame = dataset['frame']
photon_raw = dataset['photons']
bg = dataset['bg']

x = dataset['x'] 
y = dataset['y'] 
sx = dataset['lpx']
sy = dataset['lpy']
sd = (sx*2 + sy*2)**0.5

# Take x,y,sd values in camera subpixels
camera_px = 133

# Convert x,y,sd values from 'camera subpixels' to nanometres
xloc = x * camera_px
yloc = y * camera_px

phot_corr = photon_raw

# Build the output array
listLocalizations = np.column_stack((xloc, yloc, frame, phot_corr))

min_div = 100
# We divide the list into sub-lists of 'min_div' locs to minimize memory usage

Ntimes_min_div = int((listLocalizations[:,0].size/min_div))
truefalse_sum_roi_acum = []
listLocalizations_filtered = np.zeros((int(min_div*Ntimes_min_div),listLocalizations[1,:].size))

    
daa = np.zeros(listLocalizations[:,0].size)
frame_dif = np.zeros(listLocalizations[:,0].size)

Ntot = listLocalizations[:,0].size
    
for N in range(0, Ntimes_min_div+1):

    min_div_N = int(min_div)
    min_range = min_div_N*N
    max_range = (min_div_N*(N+1))
   
    if N == Ntimes_min_div:
        min_div_N = int(listLocalizations[:,0].size - min_div *(Ntimes_min_div))
        max_range = int(listLocalizations[:,0].size)
       
    truefalse = np.zeros((min_div_N,min_div_N))
    # This matrix relates each localization with the rest of localizations.
    # It will take a value of 1 if the molecules i and j (i = row, j = column)
    # are located at distances < max_dist, and are detected in frames N and (N+1)
    # or (N-1). In any other case, it will take a value of 0.

    max_dist = 20 # Value in nanometers
    
    for i in range(min_range, max_range):
        for j in range(min_range, max_range):
            daa[j-min_div_N*(N)] = ((xloc[i]-xloc[j])**2+(yloc[i]-yloc[j])**2)**(1/2)
            frame_dif[j-min_div_N*(N)] = ((listLocalizations[i,2] - listLocalizations[j,2])**2)**(1/2)
            if daa[(j-min_div_N*(N))] < max_dist and frame_dif[(j-min_div_N*(N))] == 1:
                truefalse[(i-min_range),(j-min_range)] = 1
   
    truefalse_sum = truefalse.sum(axis=0)  
    # For each row (i.e. each molecule) we calculate the sum of every
    # column from the 'truefalse' matrix.
   
    truefalse_sum_roi_acum = np.append(truefalse_sum_roi_acum, truefalse_sum)
   
idx_filtered = np.where(truefalse_sum_roi_acum > 1)
# We choose the indexes of those rows whose columnwise sum is > or = to 2

#%% APPLYING FILTER TO THE ORIGINAL LIST 

x_idx = listLocalizations[idx_filtered,0].T
y_idx = listLocalizations[idx_filtered,1].T
frame_idx = listLocalizations[idx_filtered,2].T
photons_idx = listLocalizations[idx_filtered,3].T
