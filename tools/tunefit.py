# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:26:18 2021

@author: Lucia
"""

           
## Read csv file
filename = self.tunefilename
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

angles = np.arange(self.angletune, self.angletune + self.rangeangle, self.stepsangle)
alphas = np.arange(self.alphatune, self.alphatune + self.rangealpha, self.stepsalpha)
Nang = len(angles)
Nalp = len(alphas)

D = np.ones((Nang, Nalp))

for angle in angles:
    for alpha in alphas:
            
        # The values used for the axial positions calculation of the known
        # structures through SIMPLER are obtained from the 'run_SIMPLER'
        # interface. It is important to set them correctly before running the
        # operation.
        
        self.angle = angle
        self.alpha = alpha    
        [dF, alphaF] = self.getParameters_SIMPLER()
        
           
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
        
        median_perc90_10_center = lambda x: np.mean([np.percentile(x,90), np.percentile(x,10)])
        
        # To calculate the 'median' values, we use valid localizations, 
        # i.e. those with axial positions different from 0;
        # there are elements filled with z = 0 and lateral = 0 in the 'data' matrix,
        # because not every known structures have the same number of localizations
        
        for i in np.arange(np.shape(lateral_matrix)[1]):
            c = np.where(axial_matrix[:,i]!=0) 
            photons_matrix[:,i] = self.N0*(alphaF*np.exp(-axial_matrix[:,i]/dF)
            + 1-alphaF)
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
        axial_median = (np.log(alphaF*self.N0)-np.log(photons_median
                        -(1-alphaF)*self.N0))/(1/dF)
        
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
        axial = dF*(np.log(alphaF)-np.log(photons/self.N0-(1-alphaF)))

        lateralc = lateral - lateral_median
        axialc = axial - axial_median
            
        xy = np.vstack((lateralc, axialc)).T
        circle = circle_fit.least_squares_circle(xy)
        xc = circle[0]
        yc = circle[1]
        Rc = circle[2]
        dc = 2*Rc
        
        D[angle, alpha] = dc
            

            
            
        