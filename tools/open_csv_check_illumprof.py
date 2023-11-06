# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:56:58 2022

@author: Lucia
"""


import numpy as np
import h5py as h5
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from matplotlib import cm
import time

os.chdir(r'C:\Users\Lucia\Documents\Lu Lopez\SIMPLER\2 color SIMPLER\calibraciones\220407 - SIMPLER')

t0 = time.time()
# Define filename
filename = "568_1en100_EM30_50mW_cropped_locs.hdf5"

# Read H5 file
f = h5.File(filename, "r")
dataset = f['locs']

# Load  input HDF5 file
frame = dataset['frame']
photon_raw = dataset['photons']
bg = dataset['bg']

x = dataset['x'] 
y = dataset['y'] 




# Take x,y,sd values in camera subpixels
camera_px = 133

# Convert x,y,sd values from 'camera subpixels' to nanometres
xloc = x * camera_px
yloc = y * camera_px

# CORRECT PHOTON COUNTS

# To perform this correction, the linear dependency between local laser
# power intensity and local background photons is used. Photons are
# converted to the value they would have if the whole image was illuminated
# with the maximum laser power intensity. This section is executed if the
# user has chosen to perform correction due to non-flat illumination.
#
#filename_csv = 'excitation_profile_fab.csv'
#
#datacalib = pd.read_csv(filename_csv, header=None)
#profiledata = pd.DataFrame(datacalib)
#profile = profiledata.values
#
#
#phot = photon_raw
#max_bg = np.percentile(profile, 97.5)
#phot_corr = np.zeros(photon_raw.size)