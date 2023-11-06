# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:29:41 2023

@author: Lucia
"""

import numpy as np
import h5py as h5
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from scipy.spatial.distance import cdist

os.chdir(r'C:\Users\Lucia\Documents\Lu Lopez\SIMPLER\data Chechu jun23')

# Define filename
filename = "SIMPLER1_100pM_9meratto655_2nM_100ms_647nm_130mW_telescope_12.3mm_1_MMStack_Default.ome_locs.hdf5"

# Read H5 file
f = h5.File(filename, "r")
dataset = f['locs']

# Load input HDF5 file
frame = dataset['frame']
photon_raw = dataset['photons']
bg = dataset['bg']

x = dataset['x']
y = dataset['y']
sx = dataset['lpx']
sy = dataset['lpy']
sd = (sx * 2 + sy * 2) ** 0.5

# Take x, y, sd values in camera subpixels
camera_px = 133

# Convert x, y, sd values from 'camera subpixels' to nanometers
xloc = x * camera_px
yloc = y * camera_px

phot_corr = photon_raw

# Build the output array
listLocalizations = np.column_stack((xloc, yloc, frame, phot_corr))


# Define parameters
min_div = 100
max_dist = 20  # Value in nanometers

# Calculate the number of sub-divisions
n_subdivisions = int(np.ceil(listLocalizations.shape[0] / min_div))

# Initialize the filtered indices
filtered_indices = []

# Iterate over the sub-divisions
for n in range(n_subdivisions):
    start_idx = n * min_div
    end_idx = (n + 1) * min_div
    
    if end_idx > listLocalizations.shape[0]:
        end_idx = listLocalizations.shape[0]

    # Extract the relevant subset of data for the current sub-division
    subset = listLocalizations[start_idx:end_idx]

    # Calculate pairwise distances between all localizations in the subset
    dist_matrix = cdist(subset[:, :2], subset[:, :2])

    # Calculate frame differences between all localizations in the subset
    frame_diffs = np.abs(subset[:, 2, np.newaxis] - subset[:, 2])

    # Create a boolean mask indicating localizations that satisfy the distance and frame difference conditions
    mask = np.logical_and(dist_matrix < max_dist, frame_diffs == 1)

    # Count the number of valid connections for each localization
    sum_mask = mask.sum(axis=0)

    # Filter the indices of localizations that have more than one valid connection
    filtered_indices.extend(np.where(sum_mask > 1)[0] + start_idx)

# Filter the original array based on the selected indices
listLocalizations_filtered = listLocalizations[filtered_indices]

x = listLocalizations_filtered[:,0].T
y = listLocalizations_filtered[:,1].T
photons = listLocalizations_filtered[:,3].T
framef = listLocalizations_filtered[:,2].T

x = x.flatten()
y = y.flatten()


