# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:58:41 2021

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

os.chdir(r'C:\Users\Lucia\Documents\GitHub\SIMPLER-master_Python\Example data')

t0 = time.time()
# Define filename
filename = "example_Fab.hdf5"

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
filename_csv = 'fab.csv'
#
datacalib = pd.read_csv(filename_csv, header=None)
profiledata = pd.DataFrame(datacalib)
profile = profiledata.values
#
#
phot = photon_raw
max_bg = np.percentile(profile, 97.5)
phot_corr = np.zeros(photon_raw.size)
#
#
## Correction loop
#
profx = np.size(profile,1) 
profy = np.size(profile,0) 
#
xdata = x
ydata = y
#
for i in np.arange(len(phot)):
    
    if int((np.ceil(xdata[i]))) < profx and int((np.ceil(ydata[i]))) < profy:
        phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.ceil(ydata[i])),int(np.ceil(xdata[i]))])
    elif int((np.ceil(xdata[i]))) > profx and int((np.ceil(ydata[i]))) < profy:
        phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.floor(ydata[i])),int(np.ceil(xdata[i]))])
    elif int((np.ceil(xdata[i]))) < profx and int((np.ceil(ydata[i]))) > profy:
        phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.ceil(ydata[i])),int(np.floor(xdata[i]))])
    elif int((np.ceil(xdata[i]))) > profx and int((np.ceil(ydata[i]))) > profy:
        phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.floor(ydata[i])),int(np.floor(xdata[i]))])
    
# phot_corr = photon_raw
#t1 = time.time()        
# Build the output array
listLocalizations = np.column_stack((xloc, yloc, frame, phot_corr))

# REMOVING LOCS WITH NO LOCS IN i-1 AND i+1 FRAME

# We keep a molecule if there is another one in the previous and next frame,
# located at a distance < max_dist, where max_dist is introduced by user
# (20 nm by default).
    
min_div = 1000.0
# We divide the list into sub-lists of 'min_div' locs to minimize memory usage

Ntimes_min_div = int((listLocalizations[:,0].size/min_div))
truefalse_sum_roi_acum = []
listLocalizations_filtered = np.zeros((int(min_div*Ntimes_min_div),listLocalizations[1,:].size))

    
daa = np.zeros(listLocalizations[:,0].size)
frame_dif = np.zeros(listLocalizations[:,0].size)
    
for N in range(0, Ntimes_min_div+1):
    print(N)
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

t2 = time.time() 
  
idx_filtered = np.where(truefalse_sum_roi_acum > 1)
# We choose the indexes of those rows whose columnwise sum is > or = to 2

# APPLYING FILTER TO THE ORIGINAL LIST 

x_idx = listLocalizations[idx_filtered,0].T
y_idx = listLocalizations[idx_filtered,1].T
frame_idx = listLocalizations[idx_filtered,2].T
photons_idx = listLocalizations[idx_filtered,3].T

# For the "N0 Calibration" operation, there is no "Z calculation",  
# because the aim of this procedure is to obtain N0 from a sample which 
# is supposed to contain molecules located at z ~ 0.
xl = np.array([np.amax(x_idx), np.amin(x_idx)]) 
yl = np.array([np.amax(y_idx), np.amin(y_idx)]) 
c = np.arange(0,np.size(x_idx))

hist, bin_edges = np.histogram(photons_idx[c], bins = 40, density = False)
bin_limits = np.array([bin_edges[0], bin_edges[-1]])
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

# Gaussian fit of the N0 distribution
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
A0 = np.max(hist)
mu0 = bin_centres[np.argmax(hist)]
#mu0 = np.mean(photons_idx[c])
sigma0 = np.std(photons_idx[c])
p0 = [A0, mu0, sigma0]
coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0) 

# exp fit of the N0 distribution
#def expf(x, *p):
#    a, b, d = p
#    return a*np.exp(-b*(x-d))
#
## p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
#a0 = np.max(hist)/2
#b0 = np.mean(hist)
#d0 = b0/2
#p0 = [a0, b0, d0]
#coeff, var_matrix = curve_fit(expf, bin_centres[1:], hist[1:], p0=p0)   

# Get the fitted curve
N0c = np.arange(bin_edges[1], bin_edges[-1], 100)
hist_fit = gauss(N0c, *coeff)
plt.figure()
plt.plot(bin_centres, hist, label='Non-fit data')
#plt.figure()
plt.hist(photons_idx[c], bins = 40, color = "skyblue", ec="gray")
plt.plot(N0c, hist_fit, label='Fitted data')
plt.legend()
#plt.show()