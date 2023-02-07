# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:03:00 2021

@author: Lucia

GUI for MPS x,y,z data exploration and quick analysis

conda command for converting QtDesigner file to .py:
pyuic5 -x data_explorer.ui -o data_explorer.py
    
"""

import os
cdir = os.getcwd()
os.chdir(cdir)

import ctypes
import h5py as h5
import pandas as pd
from tkinter import Tk, filedialog
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from skimage.morphology import square, dilation, disk


import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot

import data_explorer
from matplotlib import cm


# see https://stackoverflow.com/questions/1551605
# /how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105
# to understand why you need the preceeding two lines
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class MPS_explorer(QtGui.QMainWindow):
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        
        self.ui = data_explorer.Ui_MainWindow()
        self.ui.setupUi(self)
        
                
        self.initialDir = r'Desktop'
        
        
        fileformat_list = ["Picasso hdf5", "ThunderStorm csv", "custom csv"]
        self.fileformat = self.ui.comboBox_fileformat
        self.fileformat.addItems(fileformat_list)
        self.fileformat.currentIndexChanged.connect(self.emit_param)
        
        
        self.browsefile = self.ui.pushButton_browsefile
        self.browsefile.clicked.connect(self.select_file)
        
        
        self.buttonxy = self.ui.radioButtonxy
        self.buttonxz = self.ui.radioButtonxz
        self.buttonyz = self.ui.radioButtonyz
        
        self.scatter = self.ui.pushButton_scatter
        self.scatter.clicked.connect(self.scatterplot)
        
        
        self.pushButton_smallROI = self.ui.pushButton_smallROI
        self.pushButton_smallROI.clicked.connect(self.updateROIPlot)
        
        # define colormap
        
        cmap = cm.get_cmap('viridis', 100)
        colors = cmap.colors
        colors = np.delete(colors, np.s_[3], axis=1)
        col = 255*colors
        self.vir = col.astype(int)
        
       
        self.brush1 = pg.mkBrush(self.vir[20])
        self.brush2 = pg.mkBrush(self.vir[40])
        self.brush3 = pg.mkBrush(self.vir[70])
        
        self.pen1 = pg.mkPen(self.vir[20])
        self.pen2 = pg.mkPen(self.vir[40])
        self.pen3 = pg.mkPen(self.vir[70])
        
        
    def emit_param(self):
            
             
        self.fileformat1 = int(self.fileformat.currentIndex())
        
        
    def select_file(self):
        try:
            root = Tk()
            root.withdraw()
            root.filenamedata = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select file')
            if root.filenamedata != '':
                self.ui.lineEdit_filename.setText(root.filenamedata)
                
        except OSError:
            pass
        
        if root.filenamedata == '':
            return
        
    def import_file(self,filename):
        
        #File Importation
        if self.fileformat1 == 0: # Importation procedure for Picasso hdf5 files.
            
            # Read H5 file
            f = h5.File(filename, "r")
            dataset = f['locs']
        
            # Load  input HDF5 file
            frame = dataset['frame']
            photon_raw = dataset['photons']
            bg = dataset['bg']          
            
            xdata = dataset['x'] 
            ydata = dataset['y'] 

        
        
        elif self.fileformat1 == 1: # Importation procedure for ThunderSTORM csv files.
            
            ## Read ThunderSTRORM csv file
            dataset = pd.read_csv(filename)
            # Extraxt headers names
            headers = dataset.columns.values
            
            # data from different columns           
            frame = dataset[headers[np.where(headers=='frame')]].values.flatten()
            xdata = dataset[headers[np.where(headers=='x [nm]')]].values.flatten() 
            ydata = dataset[headers[np.where(headers=='y [nm]')]].values.flatten()
            zdata = dataset[headers[np.where(headers=='z [nm]')]].values.flatten()
    
            
        else: # Importation procedure for custom csv files.

            # Read custom csv file
            dataset = pd.read_csv(filename)
            data = pd.DataFrame(dataset)
            dataxyz = data.values
            dataxyz = dataxyz.astype(float)
             
            # data from different columns           

            xdata = dataxyz[:,0]
            ydata = dataxyz[:,1]
            zdata = dataxyz[:,2]

            
        return xdata, ydata, zdata
    
    # def valuechange(self):
        
    #     self.pointsize = self.slider.value()
    #     self.scatterplot()
    
    def scatterplot(self):  
        
        filename = self.ui.lineEdit_filename.text()
        xdata, ydata, zdata  = self.import_file(filename)


        self.x = xdata
        self.y = ydata
        self.z = zdata 

        
        cmapz = cm.get_cmap('viridis', np.size(self.z))
        col = cmapz.colors
        col = np.delete(col, np.s_[3], axis=1)
        col = 255*col
        self.col = col

        

        scatterWidgetxy = pg.GraphicsLayoutWidget()
        plotxy = scatterWidgetxy.addPlot(title="Scatter plot (x,y)")
        plotxy.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotxy.setAspectLocked(True)

               
        xy = pg.ScatterPlotItem(self.x, self.y, pen=None,
                                brush=self.brush1, size=1)
        
      
        plotxy.addItem(xy)
        
            
        self.empty_layout(self.ui.scatterlayout)
        self.ui.scatterlayout.addWidget(scatterWidgetxy)
        
              
        npixels = np.size(self.x)
        ROIpos = (int(min(self.x)), int(min(self.y)))
        ROIextent = int(npixels/3)


        ROIpen = pg.mkPen(color='b')
        self.roi = pg.ROI(ROIpos, ROIextent, pen = ROIpen)  
        
        self.roi.setZValue(10)
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addRotateHandle([0, 0], [1, 1])                             
        plotxy.addItem(self.roi)
        
                

        histzWidget = pg.GraphicsLayoutWidget()
        histabsz = histzWidget.addPlot(title="z Histogram")
        
        histz, bin_edgesz = np.histogram(self.z, bins=500)
        widthzabs = np.mean(np.diff(bin_edgesz))
        bincentersz = np.mean(np.vstack([bin_edgesz[0:-1],bin_edgesz[1:]]), axis=0)
        bargraphz = pg.BarGraphItem(x = bincentersz, height = histz, 
                                    width = widthzabs, brush = self.brush1, pen = None)
        histabsz.addItem(bargraphz)
                
        self.empty_layout(self.ui.zhistlayout)
        self.ui.zhistlayout.addWidget(histzWidget)
        
    
    def updateROIPlot(self):
        
                
        scatterWidgetROI = pg.GraphicsLayoutWidget()
        plotROI = scatterWidgetROI.addPlot(title="Scatter plot ROI selected")
        plotROI.setAspectLocked(True)
        
        xmin, ymin = self.roi.pos()
        xmax, ymax = self.roi.pos() + self.roi.size()
        
        
        indx = np.where((self.x>xmin) & (self.x<xmax))
        indy = np.where((self.y>ymin) & (self.y<ymax))
        
        mask = np.in1d(indx, indy)
        
        ind = np.nonzero(mask)
        index=indx[0][ind[0]]
        
        xroi = self.x[index]
        yroi = self.y[index]
        zroi = self.z[index]
        
        if self.buttonxy.isChecked():
            self.selected = pg.ScatterPlotItem(xroi, yroi, pen = self.pen1,
                                               brush = None, size = 5)  
            plotROI.setLabels(bottom=('x [nm]'), left=('y [nm]'))
            
        if self.buttonxz.isChecked():
            self.selected = pg.ScatterPlotItem(xroi, zroi, pen=self.pen2,
                                               brush = None, size = 5)
            plotROI.setLabels(bottom=('x [nm]'), left=('z [nm]'))
            
        if self.buttonyz.isChecked():
            self.selected = pg.ScatterPlotItem(yroi, zroi, pen=self.pen3,
                                               brush = None, size = 5)
            plotROI.setLabels(bottom=('y [nm]'), left=('z [nm]'))
        
        else:
            pass
        
        
        plotROI.addItem(self.selected)
        
        self.empty_layout(self.ui.scatterlayout_3)
        self.ui.scatterlayout_3.addWidget(scatterWidgetROI)    
        
        
        histzWidget2 = pg.GraphicsLayoutWidget()
        histabsz2 = histzWidget2.addPlot(title="z ROI Histogram")
        
        histz2, bin_edgesz2 = np.histogram(zroi, bins='auto')
        widthzabs2 = np.mean(np.diff(bin_edgesz2))
        bincentersz2 = np.mean(np.vstack([bin_edgesz2[0:-1],bin_edgesz2[1:]]), axis=0)
        bargraphz2 = pg.BarGraphItem(x = bincentersz2, height = histz2, 
                                    width = widthzabs2, brush = self.brush1, pen = None)
        histabsz2.addItem(bargraphz2)
                
        self.empty_layout(self.ui.zhistlayout_2)
        self.ui.zhistlayout_2.addWidget(histzWidget2)
  
        
    def empty_layout(self, layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)
    
if __name__ == '__main__':
    
    
    app = QtGui.QApplication([])
    win = MPS_explorer()
    win.show()
    app.exec_()
    

    

