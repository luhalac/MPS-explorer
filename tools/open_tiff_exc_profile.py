# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:20:58 2022

@author: Lucia
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r'C:\Users\Lucia\Documents\Lu Lopez\SIMPLER\2 color SIMPLER\exc profile\532 nm')

# Define filename
tiff_file532 = "532_SNAP_wf68x68_16_256.tif"
#tiff_file642 = "642_SNAP_wf68x68_15_256.tif"


# load tiff as array
I532 = plt.imread(tiff_file532)
#I642 = plt.imread(tiff_file642)

np.savetxt('exc_prof532.csv', I532, delimiter=',')
#np.savetxt('exc_prof642.csv', I642, delimiter=',')