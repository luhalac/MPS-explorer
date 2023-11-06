# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:43:32 2021

@author: Lucia
"""
import numpy as np
import pandas as pd
import circle_fit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import Get_Parameters as GP
import circlefit

N0 = 50000
alpha_ori = 0.9
angle_ori = 69.5

alpha = 0.9
angle = 67
lambda_exc = 642
lambda_em = 700
nI = 1.516
nS = 1.33
NA = 1.42


alphaF_ori,dF_ori = GP.getParameters_SIMPLER(lambda_exc,NA,lambda_em,angle_ori,nS,nI,alpha_ori)
alphaF,dF = GP.getParameters_SIMPLER(lambda_exc,NA,lambda_em,angle,nS,nI,alpha)

os.chdir(r'C:\Users\Lucia\Documents\NanoFísica\SIMPLER\SIMPLER-master_MATLAB\SIMPLER-master\Example data')

# Read csv file containing (lateral,axial) positions of known
# structures.

# Define filename
filename = "example_mt_rz_tunning_8mts.csv"
## Read csv file
dataset = pd.read_csv(filename, header=None)

# Lateral positions are obtained from odd columns
lateral_matrix = dataset.values[:, ::2]
lateral_matrix[np.where(np.isnan(lateral_matrix))]=0
# Axial positions are obtained from even columns
axial_matrix = dataset.values[:, 1::2]
axial_matrix[np.where(np.isnan(axial_matrix))]=0

   
# reshape into vectors    
lateral = lateral_matrix.flatten() 
axial = axial_matrix.flatten()

   
# The number of photons for each localization are retrieved from the 
# axial position and the dF, alphaF and N0 values obtained in the
# above step.

photons_matrix = np.zeros(np.shape(axial_matrix))
photons_median_matrix = np.zeros(np.shape(axial_matrix))
lateral_median_matrix = np.zeros(np.shape(axial_matrix))

# The next function allows to obtain a custom 'median' value, which is
# calculated as the mean value between the p-centile 10% and p-centile
# 90% from a given distribution. We use this function in order to
# re-center the localizations from the known structures around [0,0]



median_perc90_10_center = lambda x: np.mean([np.percentile(x,90), 
                                             np.percentile(x,10)])

# To calculate the 'median' values, we use valid localizations, 
# i.e. those with axial positions different from 0;
# there are elements filled with z = 0 and lateral = 0 in the 'data' matrix,
# because not every known structures have the same number of localizations

for i in np.arange(np.shape(lateral_matrix)[1]):
    c = np.where(axial_matrix[:,i]!=0) 
    photons_matrix[:,i] = N0*(alphaF_ori*np.exp(-axial_matrix[:,i]/dF_ori)
    + 1-alphaF_ori)
    photons_median_matrix[:,i] = (np.ones((np.shape(photons_matrix)[0]))
                                *median_perc90_10_center(photons_matrix[c,i]))
    lateral_median_matrix[:,i] = (np.ones((np.shape(lateral_matrix)[0]))
                                *median_perc90_10_center(lateral_matrix[c,i]))

# Number of photons for each lozalization    
photons = photons_matrix.flatten()
# Median value of the number of photons for the structure to which
# each localization belongs
photons_median = photons_median_matrix.flatten()
# Lateral positions median value for the structure to which
# each localization belongs                                                
lateral_median = lateral_median_matrix.flatten()
# Median value of the axial position for the structure to which 
# each localization belongs                                            
axial_median = dF*(np.log(alphaF*N0)-np.log(photons_median
                -(1-alphaF)*N0))

# Some elements from the axial vector are zero because the 'data' matrix
# contains structures with different number of localizations and thus
# there are columns filled with zeros. Now, we remove those elements. 
    
c = np.where(axial == 0)

photons = np.delete(photons,c) 
photons_median = np.delete(photons_median,c)
lateral_median = np.delete(lateral_median,c)
lateral = np.delete(lateral,c)
axial_median = np.delete(axial_median,c)
axial = np.delete(axial,c)
axial = dF*(np.log(alphaF*N0)-np.log(photons-(1-alphaF)*N0))


# The output for this function will contain the information of the
# parameters used to retrieve the number of photons from the axial
# positions; the number of photons and lateral position of every
# localization; and the median values for both the number of photons and
# the lateral positions.
# These outputs will be used by the 'Update Scatter' buttom as
# input information, in order to recalculate every axial position when
# either the angle or alpha (or both) are changed.
 

axialc = axial-axial_median
lateralc = lateral - lateral_median
# plt.scatter(lateralc,axialc)
# plt.xlabel('Lateral (nm)');
# plt.ylabel('Axial (nm)'); 
# plt.title('Combined structures');

xy = np.vstack((lateralc, axialc)).T
circle = circlefit.CircleFit(xy)
xc = circle[0]
yc = circle[1]
Rc = circle[2]
dc = 2*Rc

plotcircle = circle_fit.plot_data_circle(lateralc, axialc, xc, yc, Rc)


