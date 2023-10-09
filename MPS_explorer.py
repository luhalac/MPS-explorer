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
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from sklearn.neighbors import KDTree, NearestNeighbors
import scipy as sp


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
        
        
        self.browsefile = self.ui.pushButton_browsefile
        self.browsefile.clicked.connect(self.select_file)
        
        
        self.buttonxy = self.ui.radioButtonxy
        self.buttonxz = self.ui.radioButtonxz
        self.buttonyz = self.ui.radioButtonyz
        
        self.scatter = self.ui.pushButton_scatter
        self.scatter.clicked.connect(self.scatterplot)
        
        
        self.pushButton_smallROI = self.ui.pushButton_smallROI
        self.pushButton_smallROI.clicked.connect(self.updateROIPlot)
        
        
        # define zrange from zhist
        
        self.pushButton_zrange = self.ui.pushButton_zrange
        self.pushButton_zrange.clicked.connect(self.updateROIPlot)
        
        # perform clustering analysis (DBSCAN)
                
        self.pushButton_DBSCAN = self.ui.pushButton_DBSCAN
        self.pushButton_DBSCAN.clicked.connect(self.cluster_DBSCAN)
        
        # calculate distances between cluster centers
                
        self.pushButton_Distances = self.ui.pushButton_Distances
        self.pushButton_Distances.clicked.connect(self.dist_cmDBSCAN)
        
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
        
        self.fileformat1 = int(self.fileformat.currentIndex())

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
            zdata = dataset['z'] 
            
            # define px size in nm
            self.pxsize = 133
            
            # Convert x,y values from 'camera subpixels' to nanometres
            xdata = xdata * self.pxsize
            ydata = ydata * self.pxsize

        
        
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
        
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        
        ymin = np.min(self.y)
        ymax = np.max(self.y)

        
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
        plotxy.setXRange(xmin,xmax, padding=0)
        plotxy.setYRange(ymin,ymax, padding=0)
        
            
        self.empty_layout(self.ui.scatterlayout)
        self.ui.scatterlayout.addWidget(scatterWidgetxy)
        
              
        npixels = np.size(self.x)
        ROIpos = (int(min(self.x)), int(min(self.y)))
        ROIextent = int(npixels/10)


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
        
        print(self.roi.pos())
        
        print(self.roi.angle())
        
        xmin, ymin = self.roi.pos()
        xmax, ymax = self.roi.pos() + self.roi.size()
        
        indx = np.where((self.x>xmin) & (self.x<xmax))
        indy = np.where((self.y>ymin) & (self.y<ymax))
        
        mask = np.in1d(indx, indy)
        
        ind = np.nonzero(mask)
        index=indx[0][ind[0]]
        
        self.xroi = self.x[index]
        self.yroi = self.y[index]
        
        zmin = self.ui.lineEdit_zmin.text()
        zmax = self.ui.lineEdit_zmax.text()
        
        if zmin == "":
            
            self.zmin = None
        else:    
            self.zmin = int(self.ui.lineEdit_zmin.text())
            
        if zmax == "":
            
            self.zmax = None
        else:    
            self.zmax = int(self.ui.lineEdit_zmax.text())
        
        
        if self.zmax == None: 
            
            self.zroi = self.z[index]

        else:
            zroi = self.z[index]
            indz = np.where((zroi>self.zmin) & (zroi<self.zmax))
            
            self.zroi = zroi[indz]
            self.xroi = self.xroi[indz]
            self.yroi = self.yroi[indz]
            
        
        
        if self.buttonxy.isChecked():
            self.selected = pg.ScatterPlotItem(self.xroi, self.yroi, pen = self.pen1,
                                               brush = None, size = 5)  
            plotROI.setLabels(bottom=('x [nm]'), left=('y [nm]'))
            plotROI.setXRange(np.min(self.xroi), np.max(self.xroi), padding=0)
            
        if self.buttonxz.isChecked():
            self.selected = pg.ScatterPlotItem(self.zroi, self.xroi, pen=self.pen2,
                                               brush = None, size = 5)
            plotROI.setLabels(bottom=('z [nm]'), left=('x [nm]'))
            plotROI.setXRange(np.min(self.zroi), np.max(self.zroi), padding=0)
            
        if self.buttonyz.isChecked():
            self.selected = pg.ScatterPlotItem(self.zroi, self.xroi, pen=self.pen3,
                                               brush = None, size = 5)
            plotROI.setLabels(bottom=('z [nm]'), left=('x [nm]'))
            plotROI.setXRange(np.min(self.zroi), np.max(self.zroi), padding=0)
        
        else:
            pass
        
        
        plotROI.addItem(self.selected)
        
        self.empty_layout(self.ui.scatterlayout_3)
        self.ui.scatterlayout_3.addWidget(scatterWidgetROI)    
        
        self.ui.scatterlayout_3.addWidget(scatterWidgetROI)    
        
        
        histzWidget2 = pg.GraphicsLayoutWidget()
        histabsz2 = histzWidget2.addPlot(title="z ROI Histogram")
        
        histz2, bin_edgesz2 = np.histogram(self.zroi, bins='auto')
        widthzabs2 = np.mean(np.diff(bin_edgesz2))
        bincentersz2 = np.mean(np.vstack([bin_edgesz2[0:-1],bin_edgesz2[1:]]), axis=0)
        bargraphz2 = pg.BarGraphItem(x = bincentersz2, height = histz2, 
                                    width = widthzabs2, brush = self.brush1, pen = None)
        histabsz2.addItem(bargraphz2)
                
        self.empty_layout(self.ui.zhistlayout_2)
        self.ui.zhistlayout_2.addWidget(histzWidget2)
        
        

    def cluster_DBSCAN(self):
        
        X = np.column_stack((self.xroi,self.yroi,self.zroi))
        
        self.eps = float(self.ui.lineEdit_eps.text())
        self.minsamples = float(self.ui.lineEdit_minsamples.text())
        
        min_samples = int(self.minsamples)
        eps = int(self.eps)

        
        db = DBSCAN(eps = eps, min_samples = min_samples).fit(X) 
        dblabels = db.labels_
        
        # compute mass centers of the clusters
        cm_list = [] 
       
        for i in range(np.max(dblabels)):
            idx = np.where(dblabels==i)
            x_i = self.xroi[idx]
            y_i = self.yroi[idx]
            z_i = self.zroi[idx]
            cm_list.append(np.array([np.mean(x_i),np.mean(y_i),np.mean(z_i)]))
            
        # Remove the noise
        range_max = len(X)
        Xc = np.array([X[i] for i in range(0, range_max) if dblabels[i] != -1])
        labels = np.array([dblabels[i] for i in range(0, range_max) if dblabels[i] != -1])
        
        cmapz = cm.get_cmap('viridis', np.size(labels))
        col = cmapz.colors
        col = np.delete(col, np.s_[3], axis=1)
        col = 255*col
       

        self.cms = np.array(cm_list)
            
        scatterWidgetDBSCAN = pg.GraphicsLayoutWidget()
        plotclusters = scatterWidgetDBSCAN.addPlot(title="Clustered data with DBSCAN")
        plotclusters.setAspectLocked(True)

        self.selecteddata = pg.ScatterPlotItem(X[:,0], X[:,1], size=2)  
        self.selectedcluscm = pg.ScatterPlotItem(self.cms[:,0], self.cms[:,1], size=10)  
        self.selectedclus = pg.ScatterPlotItem(Xc[:,0], Xc[:,1], pen=[pg.mkPen(v) for v in col],brush=pg.mkBrush(None), size=10) 
        plotclusters.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotclusters.setXRange(np.min(self.xroi), np.max(self.xroi), padding=0)
        
        plotclusters.addItem(self.selecteddata)
        plotclusters.addItem(self.selectedclus)
        plotclusters.addItem(self.selectedcluscm)
        
        self.empty_layout(self.ui.scatterlayout_dbscan)
        self.ui.scatterlayout_dbscan.addWidget(scatterWidgetDBSCAN)    
        
    def dist_cmDBSCAN(self):
        
        # compute distances to nearest neighbors (cm of the clusters obtained with DBSCAN)
        self.Nneighbor = float(self.ui.lineEdit_Nneighbor.text())
        Nneighbor = int(self.Nneighbor)
        print(Nneighbor)

        tree = KDTree(self.cms)
        distances, indexes = tree.query(self.cms, Nneighbor+1) 
        distances = distances[:,1:] # exclude distance to the same molecule; distances has N rows (#clusters) and M columns (# neighbors)
        indexes = indexes[:,1:]    
        
        scatterWidgetDBSCAN_cmdist = pg.GraphicsLayoutWidget()
        plotdistcm = scatterWidgetDBSCAN_cmdist.addPlot(title="Clusters centers and distances")
        plotdistcm.setAspectLocked(True)

        self.selectedcluscm = pg.ScatterPlotItem(self.cms[:,0], self.cms[:,1], size=10)  
        plotdistcm.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotdistcm.setXRange(np.min(self.xroi), np.max(self.xroi), padding=0)
        
        plotdistcm.addItem(self.selectedcluscm)
        
        self.empty_layout(self.ui.scatterlayout_cmdbscan)
        self.ui.scatterlayout_cmdbscan.addWidget(scatterWidgetDBSCAN_cmdist) 
        
        
        

  
        
    def empty_layout(self, layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)
    
if __name__ == '__main__':
    
    
    app = QtGui.QApplication([])
    win = MPS_explorer()
    win.show()
    app.exec_()
    

    

