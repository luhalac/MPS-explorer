# GET PARAMETERS
import numpy as np
from scipy.interpolate import interp1d
import scipy.optimize as opt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

dF = []
alphaF = []
dif = []

def getParameters_SIMPLER(lambda_exc,NA,lambda_em,angle,nS,nI,alpha):
    
    # Angle
    if angle == 0:
        angle = np.arcsin(NA/nI)
    else:
        angle = np.deg2rad(angle)
        
    # Alpha
    if alpha == 0:
        alpha = 0.9
    else:
        alpha = alpha
        
    # Z
    z_fit = np.arange(5, 500, 0.5)
    z =[5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    
    # Lambda
    lambda_em_max =[500, 530, 560, 590, 620, 670, 700, 720]
    dif = np.min(abs(lambda_em_max - np.ones(np.size(lambda_em_max))*lambda_em))
    i_lambda_em = np.argmin(abs(lambda_em_max - np.ones(np.size(lambda_em_max))*lambda_em))

    
    # Axial dependece of the excitation field
    d = lambda_exc/(4 * np.pi * np.sqrt(nI**2*(np.sin(angle)**2) - nS**2))
    I_exc = alpha * np.exp(-z_fit/d) + (1-alpha)
    
    # Axial dependece of the fraction of fluorescence collected by a microscope objetive
    if NA == 1.42:
        DF_NA1_42 = np.loadtxt('DF_NA1.42.txt')
        DFi = DF_NA1_42[:,i_lambda_em]
    elif NA == 1.45:
        DF_NA1_45 = np.loadtxt('DF_NA1.45.txt')
        DFi = DF_NA1_45[:,i_lambda_em]
    elif NA == 1.49:
        DF_NA1_49 = np.loadtxt('DF_NA1.49.txt')
        DFi = DF_NA1_49[:,i_lambda_em]

    DFi_interp = interp1d(z,DFi)(z_fit)
    I_total = I_exc * DFi_interp
    #plt.plot(z, DFi, 'o', z_fit, DFi_interp, '-')
    #plt.show()
       
    # Fit F to "F = alphaF*exp(-z/dF)+(1-alphaF) 
    def fitF(x, a, b, c):
        return a * np.exp(-b*x) + c
    
                                 
    popt, pcov = curve_fit(fitF, z_fit, I_total, p0 = [0.9, 0.01, 0.05])
    
    #     # check the goodness of fit by plotting
    # plt.plot(z_fit, I_total, 'b-', label='data')
    # plt.plot(z_fit, fitF(z_fit, *popt), 'r-',
    #      label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    
    alphaF = 1-popt[2]
    dF = 1/popt[1]
    
    return alphaF, dF
    
    
    
    
    
    
    
    
    
    
    
    
    
