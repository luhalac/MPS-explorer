# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:33:28 2021

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
from scipy.ndimage import gaussian_filter

os.chdir(r'C:\Users\Lucia\Documents\GitHub\SIMPLER-master_Python\Example data')


# Define filename
filename = "example_npc.hdf5"

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

min_div = 100.0
# We divide the list into sub-lists of 'min_div' locs to minimize memory usage

Ntimes_min_div = int((listLocalizations[:,0].size/min_div))
truefalse_sum_roi_acum = []
listLocalizations_filtered = np.zeros((int(min_div*Ntimes_min_div),listLocalizations[1,:].size))

    
daa = np.zeros(listLocalizations[:,0].size)
frame_dif = np.zeros(listLocalizations[:,0].size)
    
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


#%% Z-Calculation

alphaF = 0.96
N0 = 10000
dF = 87.7
photons1 = photons_idx

# Small ROI and Large ROI cases


z1 = (np.log(alphaF*N0)-np.log(photons1-(1-alphaF)*N0))/(1/dF)
z1 = np.real(z1)

# Scatter plots / Projection r vs. z, x vs. z , y vs. z
# ------------------------------------------------------------------------
# If the selected operation is "(r,z) Small ROI", we perform a first step 
# where the main axis is obtained from a linear fit of the (x,y) data

x = x_idx.flatten()
y = y_idx.flatten()
z = z1.flatten()

M = np.matrix([x,y,z])
P = np.polyfit(x,y,1)
yfit = P[0]*x + P[1]
def Poly_fun(x):
    y_polyfunc = P[0]*x + P[1]
    return y_polyfunc

Origin_X = 0.99999*min(x)
Origin_Y = Poly_fun(Origin_X)

# Change from cartesian to polar coordinates
tita = np.arctan(P[0])
tita1 = np.arctan((y-Origin_Y)/(x-Origin_X))
r = ((x-Origin_X)**2+(y-Origin_Y)**2)**(1/2)
tita2 = [x - tita for x in tita1]
proyec_r = np.cos(tita2)*r



# This function builds a 2D matrix where each localizations is plotted as a
# normalized Gaussian function.

r = proyec_r

mag=100
sigma_lat=2
sigma_ax=2   



# define px size in the SR image (in nm)
pxsize_render = camera_px/mag
r = y
z = x
# re define origin for lateral and axial coordinates

# define px size in the SR image (in nm)
sigma_latpx = sigma_lat/pxsize_render
sigma_axpx = sigma_ax/pxsize_render

# re define origin for lateral and axial coordinates
r_ori = r-min(r) + sigma_lat
z_ori = z-min(z) + sigma_ax

# re define max
max_r = (max(r)-min(r)) + sigma_lat
max_z = (max(z)-min(z)) + sigma_ax

# 

SMr = r_ori/pxsize_render
SMz = z_ori/pxsize_render

sigma_latpx = sigma_lat/pxsize_render
sigma_axpx = sigma_ax/pxsize_render

# Definition of pixels affected by the list of SM (+- 5*sigma)
A = np.zeros((np.int(np.ceil(max_r/pxsize_render)), np.int(np.ceil(max_z/pxsize_render))))

for i in np.arange(len(SMr)):
    A[int(np.floor(SMr[i])), int(np.floor(SMz[i]))] = 1


sigma = np.array((sigma_latpx,sigma_axpx))

GB = gaussian_filter(A, sigma)

fig, ax = plt.subplots()
ax.imshow(GB)


sigma_width_nm = np.max([sigma_lat, sigma_ax]);
sigma_width_px = sigma_width_nm/pxsize_render;
sd = disk(int(np.round(5*sigma_width_px)));
# This matrix contains 1 in +- 5 sigma units around the SM positions
A_affected = dilation(A,sd);
# r and z positions affected
indaffected = np.where(A_affected==1)
raffected = indaffected[0]
zaffected = indaffected[1]


#'PSF' is a function that calculates the value for a given position (r,z)
# assuming a 2D Gaussian distribution centered at (SMr(k),SMz(k))
def PSF(r, z, SMr, SMz, I):
    psf = (I/(2*np.pi*sigma_latpx*sigma_axpx))*np.exp(-(((r-SMr)**2)/(2*sigma_latpx**2) + ((z-SMz)**2)/(2*sigma_axpx**2)))
    return psf

# B = empty matrix that will be filled with the values given by the
# Gaussian blur of the points listed in (SMr, SMz)
B = np.zeros((np.int(np.ceil(5*sigma_width_px+(max_r/pxsize_render))), 
              np.int(np.ceil(5*sigma_width_px+(max_z/pxsize_render)))))

# For each molecule from the list
for k in np.arange(len(SMr)):
    # For each pixel of the final image with a value different from zero
    for i in np.arange(len(raffected)): 
        B[raffected[i],zaffected[i]] = B[raffected[i],zaffected[i]] + PSF(raffected[i],zaffected[i],SMr[k],SMz[k],1)
        # Each 'affected' pixel (i) will take the value that had at the beggining
        # of the k-loop + the value given by the distance to the k-molecule.
    

fig, ax = plt.subplots()
ax.imshow(B)    





        
        
        
        
        