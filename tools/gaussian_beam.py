# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:02:24 2021

@author: Lucia
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from skimage.morphology import square, dilation, disk
from skimage.external import tifffile as tif
from skimage import io


N = 10 # number of frames
size = 400
fov_center = [0,0]
G = np.zeros((N, size,size))
posx = np.arange(0,100,N)
posy = np.arange(0,100,N)

def gaussian(r, fwhm):
    
    """ 2D gaussian beam intensity """
    
    β = 100
    I = β * np.exp(-4 * np.log(2) * (r**2/fwhm**2))
    
    return I

def psf(central_zero, size, px, fov_center, fwhm, d):
    """ 2D extension for a 1D function in (r,θ) """

    center = fov_center
    x0 = central_zero[0]
    y0 = central_zero[1]

    x = np.arange(-size/2 - center[0], size/2 - center[0], px)
    y = np.arange(-size/2 - center[1], size/2 - center[1], px)

    # CAMBIE Y POR -Y PARA NO TENER QUE HACER FLIPUD
    [Mx, My] = np.meshgrid(x, -y)
    # My = np.flipud(My)

    Mro = np.sqrt((Mx-x0)**2 + (My-y0)**2)
    result = gaussian(Mro, fwhm)
    
    

    return result

for i in np.arange(N):
    
    G[i,:,:] = psf([posx[i],posy[i]], 400, 1, [0,0], 100, 1)
    # plt.figure()
    # plt.imshow(G[i,:,:])
    
G = G.astype('int8')
tif.imsave('G.tif', G, bigtiff=True)



# read the image stack
img = io.imread('G.tif')