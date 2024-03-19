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
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from sklearn.neighbors import KDTree, NearestNeighbors
import tools.utils as utils
import hdbscan


import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets

import data_explorer



# see https://stackoverflow.com/questions/1551605
# /how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105
# to understand why you need the preceeding two lines
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class MPS_explorer(QtWidgets.QMainWindow):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = data_explorer.Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Define initial directory
        self.initialDir = "Desktop"  # You can set the initial directory here
        
        # File Formats
        fileformat_list = ["Picasso hdf5", "ThunderStorm csv", "custom csv"]
        self.fileformat = self.ui.comboBox_fileformat
        self.fileformat.addItems(fileformat_list)
        self.fileformat_2 = self.ui.comboBox_fileformat_2
        self.fileformat_2.addItems(fileformat_list)  # Reuse the same list for second channel
        
        # Connect Buttons to Methods
        self.ui.pushButton_browsefile.clicked.connect(self.select_file)
        self.ui.pushButton_browsefile_2.clicked.connect(self.select_file2)
        self.ui.pushButton_scatter.clicked.connect(self.scatterplot)
        self.ui.pushButton_zrange.clicked.connect(self.update_ROI)
        self.ui.pushButton_savexyzROI.clicked.connect(self.savexyzROI)
        self.ui.pushButton_savexyzROI_2.clicked.connect(self.savexyzROI2)
        self.ui.pushButton_savedistdata.clicked.connect(self.savedistdata)
        self.ui.pushButton_DBSCAN.clicked.connect(self.cluster)
        self.ui.pushButton_remove_bad_cluster.clicked.connect(self.dist_cm_good_clus)
        self.ui.pushButton_savecluscenters.clicked.connect(self.save_clus_CM)
        self.ui.pushButton_Distances.clicked.connect(self.KNdist_hist)
        
        # Fine Tuning Parameters
        self.ui.lineEdit_latmin.textChanged.connect(self.latchange)
        self.ui.lineEdit_latmax.textChanged.connect(self.latchange)
        self.ui.lineEdit_bin.textChanged.connect(self.latchange)
        
        self.lmin = 0
        self.lmax = 800
        self.bins = 30
        
        # Colors
        self.brush1 = pg.mkBrush("#d55e00")
        self.brush2 = pg.mkBrush("#009e73")
        self.brush3 = pg.mkBrush("#0072b2")
        
        self.pen1 = pg.mkPen("#d55e00")
        self.pen2 = pg.mkPen("#009e73")
        self.pen3 = pg.mkPen("#0072b2")
        
        # ROI Shape Radio Buttons
        self.radioButton_circROI = self.ui.radioButton_circROI
        self.radioButton_squareROI = self.ui.radioButton_squareROI
        self.radioButton_circROI.clicked.connect(self.scatterplot)
        self.radioButton_squareROI.clicked.connect(self.scatterplot)



    def select_file(self):
        try:
            root = Tk()
            root.withdraw()
            root.filenamedata = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select file')
            if root.filenamedata != '':
                self.ui.lineEdit_filename.setText(root.filenamedata)
                # Obtener el formato del archivo seleccionado
                self.fileformat1 = int(self.fileformat.currentIndex())
                # Llamar a la función import_file
                self.xdata, self.ydata, self.zdata = self.import_file(root.filenamedata, self.fileformat1)
        except OSError:
            pass
        
        if root.filenamedata == '':
            return

    def select_file2(self):
        try:
            root = Tk()
            root.withdraw()
            root.filenamedata2 = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select file')
            if root.filenamedata2 != '':
                self.ui.lineEdit_filename_2.setText(root.filenamedata2)
                # Obtener el formato del archivo seleccionado
                self.fileformat2 = int(self.fileformat_2.currentIndex())
                # Llamar a la función import_file
                self.xdata2, self.ydata2, self.zdata2 = self.import_file(root.filenamedata2,  self.fileformat2)
        except OSError:
            pass
        
        if root.filenamedata2 == '':
            return

    
    
    def import_file(self, filename, fileformat):
        if fileformat == 0: # Importation procedure for Picasso hdf5 files.
            f = h5.File(filename, "r")
            dataset = f['locs']
            xdata = dataset['x'] 
            ydata = dataset['y'] 
            zdata = dataset['z'] 
            self.pxsize = 133
            xdata = xdata * self.pxsize
            ydata = ydata * self.pxsize
        elif fileformat == 1: # Importation procedure for ThunderSTORM csv files.
            dataset = pd.read_csv(filename)
            headers = dataset.columns.values
            xdata = dataset[headers[np.where(headers=='x [nm]')]].values.flatten() 
            ydata = dataset[headers[np.where(headers=='y [nm]')]].values.flatten()
            zdata = dataset[headers[np.where(headers=='z [nm]')]].values.flatten()
        else: # Importation procedure for custom csv files.
            dataset = pd.read_csv(filename)
            data = pd.DataFrame(dataset)
            dataxyz = data.values
            dataxyz = dataxyz.astype(float)
            xdata = dataxyz[:,0]
            ydata = dataxyz[:,1]
            zdata = dataxyz[:,2]
        return xdata, ydata, zdata


    def scatterplot(self):  
        
        # Scatter plot data Ch1
        filename1 = self.ui.lineEdit_filename.text()
        xdata, ydata, zdata  = self.import_file(filename1, self.fileformat1)
 
        self.x = xdata
        self.y = ydata
        self.z = zdata 
        
        self.data_points = np.column_stack((self.x, self.y))
        
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        
        ymin = np.min(self.y)
        ymax = np.max(self.y)


        scatterWidgetxy = pg.GraphicsLayoutWidget()
        plotxy = scatterWidgetxy.addPlot(title="Scatter plot (x,y) both channels")
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
        


        ROIpen = pg.mkPen(color='r')
        
        
        if self.ui.radioButton_circROI.isChecked():
            
            # Create circular ROI
            self.circular_roi = pg.CircleROI(ROIpos, ROIextent, movable=True, pen = ROIpen)
            self.circular_roi.handleColor = (255, 0, 0)  # Set handles color to red
            self.circular_roi.addScaleHandle([1, 1], [0, 0])
            self.circular_roi.setZValue(10)
            # Add ROI to the scatterplot
            plotxy.addItem(self.circular_roi)
            # Connect signals
            self.circular_roi.sigRegionChangeFinished.connect(self.update_ROI)

            
        elif self.ui.radioButton_squareROI.isChecked():
        
            # Create square ROI
            self.square_roi = pg.ROI(ROIpos, ROIextent, pen = ROIpen)  
            self.square_roi.setZValue(10)
            self.square_roi.addScaleHandle([1, 1], [0, 0])
            self.square_roi.addRotateHandle([0, 0], [1, 1]) 
            # Add ROI to the scatterplot
            plotxy.addItem( self.square_roi)   
            # Connect signals
            self.square_roi.sigRegionChangeFinished.connect(self.update_ROI)

        else:
            pass            

    

        histzWidget = pg.GraphicsLayoutWidget()
        histabsz = histzWidget.addPlot(title="z Histogram Ch 1")
        
        histz, bin_edgesz = np.histogram(self.z, bins=500)
        widthzabs = np.mean(np.diff(bin_edgesz))
        bincentersz = np.mean(np.vstack([bin_edgesz[0:-1],bin_edgesz[1:]]), axis=0)
        bargraphz = pg.BarGraphItem(x = bincentersz, height = histz, 
                                    width = widthzabs, brush = self.brush1, pen = self.pen1)
        histabsz.addItem(bargraphz)
                
        self.empty_layout(self.ui.zhistlayoutch1)
        self.ui.zhistlayoutch1.addWidget(histzWidget)

        
        # Scatter plot data Ch2

        filename2 = self.ui.lineEdit_filename_2.text()
        
        if filename2 == '':
            
            
            pass
        
        else:

            xdata2, ydata2, zdata2  = self.import_file(filename2, self.fileformat2)
    
            self.x2 = xdata2
            self.y2 = ydata2
            self.z2 = zdata2 
    
    
            xy2 = pg.ScatterPlotItem(self.x2, self.y2, pen=None,
                                    brush=self.brush2, size=1)
            
                   
            plotxy.addItem(xy2)
    
            self.empty_layout(self.ui.scatterlayout)
            self.ui.scatterlayout.addWidget(scatterWidgetxy)
            
            histzch2Widget = pg.GraphicsLayoutWidget()
            histabszch2 = histzch2Widget.addPlot(title="z Histogram Ch 2")
            
            histz2, bin_edgesz2 = np.histogram(self.z2, bins=500)
            widthzabs2 = np.mean(np.diff(bin_edgesz2))
            bincentersz2 = np.mean(np.vstack([bin_edgesz2[0:-1],bin_edgesz2[1:]]), axis=0)
            bargraphz2 = pg.BarGraphItem(x = bincentersz2, height = histz2, 
                                        width = widthzabs2, brush = self.brush2, pen = self.pen2)
            histabszch2.addItem(bargraphz2)
                    
            self.empty_layout(self.ui.zhistlayoutch2)
            self.ui.zhistlayoutch2.addWidget(histzch2Widget)
     
              
          
    
    def update_ROI(self):
        
                        
        scatterWidgetROI = pg.GraphicsLayoutWidget()
        plotROI = scatterWidgetROI.addPlot(title="Scatter plot ROI selected")
        plotROI.setAspectLocked(True)
        
        if self.ui.radioButton_circROI.isChecked():
        # Get circular ROI position and size
            pos = self.circular_roi.pos()
            size = self.circular_roi.size()
            
            diameter = self.circular_roi.size()
    
            # Calculate and return the radius (half of the diameter)
            radius = diameter / 2
           
            # Calculate the center coordinates
            center_x = pos.x() + size / 2
            center_y = pos.y() + size / 2
            
            center = np.column_stack((center_x, center_y))
            
            
            
            # Iterate through data points and check if they are inside the circular ROI
            points_inside_roi = []
            for point in self.data_points:
                if np.any(np.linalg.norm(point - center) <= radius):
                    points_inside_roi.append(point)
            
            # Convert list of points to numpy array
            points_inside_roi = np.array(points_inside_roi)
            
            
            
            self.xroi = points_inside_roi[:,0]
            self.yroi = points_inside_roi[:,1]
            
            
            self.zroi = [self.z[i] for i, point in enumerate(self.data_points) if point in points_inside_roi]
            
            
            # Define zmin and zmax
            zmin = self.ui.lineEdit_zmin.text()
            zmax = self.ui.lineEdit_zmax.text()
            
            # Convert zmin and zmax to numeric types if they are strings
            if zmin == "":
                self.zmin = None
            else:    
                self.zmin = float(zmin)
            
            if zmax == "":
                self.zmax = None
            else:    
                self.zmax = float(zmax)
            
            
            self.zroi = np.array(self.zroi)
            # Define the z sectioning using zmin and zmax
            if self.zmin is not None and self.zmax is not None:
                z_roi_indices = np.where((self.zroi >= self.zmin) & (self.zroi <= self.zmax))[0]
                self.zroi = self.zroi[z_roi_indices]
            else:
                # Keep all z values if zmin or zmax is not defined
                pass
  
        elif self.ui.radioButton_squareROI.isChecked():
            
            # get square ROI position and size
            xmin, ymin = self.square_roi.pos()
            xmax, ymax = self.square_roi.pos() + self.square_roi.size()
        
            indx = np.where((self.x > xmin) & (self.x < xmax))
            indy = np.where((self.y > ymin) & (self.y < ymax))
            mask = np.in1d(indx, indy)
            ind = np.nonzero(mask)
            index = indx[0][ind[0]]
            self.xroi = self.x[index]
            self.yroi = self.y[index]
        
        
            zmin = self.ui.lineEdit_zmin.text()
            zmax = self.ui.lineEdit_zmax.text()
        
            self.zmin = int(zmin) if zmin else None
            self.zmax = int(zmax) if zmax else None
        
            if self.zmax is None:
                self.zroi = self.z[index]
            else:
                zroi = self.z[index]
                indz = np.where((zroi > self.zmin) & (zroi < self.zmax))
                self.zroi = zroi[indz]
                self.xroi = self.xroi[indz]
                self.yroi = self.yroi[indz]
          
                
    
        self.selected = pg.ScatterPlotItem(self.xroi, self.yroi, pen = self.pen1,
                                           brush = None, size = 5)  
        plotROI.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotROI.setXRange(np.min(self.xroi), np.max(self.xroi), padding=0)
        plotROI.addItem(self.selected)
        
        
        self.empty_layout(self.ui.scatterlayout_3)
        self.ui.scatterlayout_3.addWidget(scatterWidgetROI)    
        
        
        histzWidget2 = pg.GraphicsLayoutWidget()
        histabsz2 = histzWidget2.addPlot(title="z ROI Histogram")
        
        histz2, bin_edgesz2 = np.histogram(self.zroi, bins='auto')
        widthzabs2 = np.mean(np.diff(bin_edgesz2))
        bincentersz2 = np.mean(np.vstack([bin_edgesz2[0:-1],bin_edgesz2[1:]]), axis=0)
        bargraphz2 = pg.BarGraphItem(x = bincentersz2, height = histz2, 
                                    width = widthzabs2, brush = self.brush1, pen = self.pen1)
        bargraphz2.setOpacity(0.5) 
        histabsz2.addItem(bargraphz2)
        
                
        self.empty_layout(self.ui.zhistlayout_2)
        self.ui.zhistlayout_2.addWidget(histzWidget2)
        
           

    def savexyzROI(self):
        
        xyzROI = np.array([self.xroi,self.yroi,self.zroi])
        xyzROI = np.transpose(xyzROI)
        
        filename = self.ui.lineEdit_filename.text()
        filename = os.path.splitext(filename)[0]
        dataNamecsv = utils.insertSuffix(filename, '_xyzROICh1.csv')
        
        #export array to CSV file (using 2 decimal places)
        np.savetxt(dataNamecsv, xyzROI, delimiter=",", fmt="%.2f",header="x, y, z", comments="")

        
        
    def savexyzROI2(self):
    
        xyzROI2 = np.array([self.xroi2,self.yroi2,self.zroi2])
        xyzROI2 = np.transpose(xyzROI2)
        
        filename2 = self.ui.lineEdit_filename_2.text()
        filename2 = os.path.splitext(filename2)[0]
        dataNamecsv = utils.insertSuffix(filename2, '_xyzROICh2.csv')

        
        #export array to CSV file (using 2 decimal places)
        np.savetxt(dataNamecsv, xyzROI2, delimiter=",", fmt="%.2f",header="x, y, z", comments="")
        
    
    def savedistdata(self):
        
        Nneighbor = int(self.Nneighbor)

        dist = self.distances
        
        filename = self.ui.lineEdit_filename.text()
        filename = os.path.splitext(filename)[0]
        dataNamecsv = utils.insertSuffix(filename, '_' + str(Nneighbor) + 'neighbor_distdata.csv')
        
        #export dist array to CSV file (using 2 decimal places)
        np.savetxt(dataNamecsv, dist, delimiter=",", fmt="%.2f")

    def cluster(self):
        
        
        self.badclusterslist = []
        self.indbc = []

        
        # locs in ROI
        # XYZ = np.column_stack((self.xroi, self.yroi, self.zroi))
        
        # XY locs in ROI
        XYZ = np.column_stack((self.xroi, self.yroi))
               
        # DBSCAN parameters
        # self.eps = float(self.ui.lineEdit_eps.text())
        # eps = int(self.eps)
        
        
        self.minsamples = float(self.ui.lineEdit_minsamples.text())
        
        min_samples = int(self.minsamples)
        

        # DBSCAN without previous filtering
        # db = DBSCAN(eps = 30, min_samples = min_samples).fit(XYZ) 
        # dblabels = db.labels_
        
        db = hdbscan.HDBSCAN(min_cluster_size=min_samples, gen_min_span_tree=True).fit(XYZ)
        dblabels = db.labels_
        
        cm_list = [] 
       
        for i in range(np.max(dblabels)):
            idx = np.where(dblabels==i)
            x_i = self.xroi[idx]
            y_i = self.yroi[idx]
            # z_i = self.zroi[idx]
            cm_list.append(np.array([np.mean(x_i),np.mean(y_i)]))
        
   
        range_max = len(XYZ)
        Xc = np.array([XYZ[i] for i in range(0, range_max) if dblabels[i] != -1])
        
        self.cms = np.array(cm_list)
        self.cms = np.around(self.cms, decimals=2)

 
        scatterWidgetDBSCAN = pg.GraphicsLayoutWidget()
        plotclusters = scatterWidgetDBSCAN.addPlot(title="Clustered data")
        plotclusters.setAspectLocked(True)

        self.selecteddata = pg.ScatterPlotItem(XYZ[:,0], XYZ[:,1], size=2, brush = self.brush1)  
        self.selectedcluscm = pg.ScatterPlotItem(self.cms[:,0], self.cms[:,1], size=10, brush = self.brush3)  
        self.selectedclus = pg.ScatterPlotItem(Xc[:,0], Xc[:,1], pen=self.pen1,brush=pg.mkBrush(None), size=10) 
        
        plotclusters.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotclusters.setXRange(np.min(self.xroi), np.max(self.xroi), padding=0)
        
        plotclusters.addItem(self.selecteddata)
        plotclusters.addItem(self.selectedclus)
        plotclusters.addItem(self.selectedcluscm)
        
        self.empty_layout(self.ui.scatterlayout_dbscan)
        self.ui.scatterlayout_dbscan.addWidget(scatterWidgetDBSCAN) 
        
        scatterWidgetDBSCAN_cmdist = pg.GraphicsLayoutWidget()
        plotdistcmd = scatterWidgetDBSCAN_cmdist.addPlot(title="Clusters centers and distances")
        plotdistcmd.setAspectLocked(True)

        self.selectedcluscmd = pg.ScatterPlotItem(self.cms[:,0], self.cms[:,1], size=10, brush = self.brush3)  
        plotdistcmd.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotdistcmd.setXRange(np.min(self.xroi), np.max(self.xroi), padding=0)
        self.gcms = []
        self.selectedcluscmd.sigClicked.connect(self.rx)
        
        plotdistcmd.addItem(self.selectedcluscmd)
        
        self.empty_layout(self.ui.scatterlayout_histcmdist)
        self.ui.scatterlayout_histcmdist.addWidget(scatterWidgetDBSCAN_cmdist) 
     

 
    def rx(self, obj, points):

        # badclus = (points[0].pos())
        badcluscoord = np.array((np.round(points[0].pos()[0], decimals = 2), np.round(points[0].pos()[1], decimals = 2)))

        indbad_clus = np.where((self.cms[:,0] == badcluscoord[0]) & (self.cms[:,1] == badcluscoord[1]))

        self.indbc.append(indbad_clus)
        badind = np.unique(self.indbc)
        
        self.good_cms = [elem for i, elem in enumerate(self.cms) if i not in badind]
        self.gcms = np.array(self.good_cms)
   

        
    
    def dist_cm_good_clus(self):
        
        scatterWidgetgoodclus = pg.GraphicsLayoutWidget()
        plotgoodclus = scatterWidgetgoodclus.addPlot(title="Clusters centers and distances")
        plotgoodclus.setAspectLocked(True)
        
        
        if len(self.gcms) == 0:
            self.gcms = self.cms
        else:
            pass

        self.selectedgoodclus = pg.ScatterPlotItem(self.gcms[:,0], self.gcms[:,1], size=10, brush = self.brush3)  
        plotgoodclus.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotgoodclus.setXRange(np.min(self.xroi), np.max(self.xroi), padding=0)
        
        plotgoodclus.addItem(self.selectedgoodclus)
        
        self.empty_layout(self.ui.scatterlayout_goodclus)
        self.ui.scatterlayout_goodclus.addWidget(scatterWidgetgoodclus)
        
    def save_clus_CM(self):
        
        clusCMxy = np.array([self.gcms[:,0],self.gcms[:,1]])
        clusCMxy = np.transpose(clusCMxy)
        
        filename = self.ui.lineEdit_filename.text()
        filename = os.path.splitext(filename)[0]
        dataNamecsv = utils.insertSuffix(filename, '_clusCM_xy.csv')
        
        #export array to CSV file (using 2 decimal places)
        np.savetxt(dataNamecsv, clusCMxy, delimiter=",", fmt="%.2f", comments="")
        
        
        
        
    def latchange(self):
        
        self.bins = int(self.nbins.text())
        self.lmin = float(self.latmin.text())
        self.lmax = float(self.latmax.text())
        
        print(self.lmax)
        
      
        self.KNdist_hist()
        
    def KNdist_hist(self):
        
        # compute distances to nearest neighbors (cm of the clusters obtained with DBSCAN)
        self.Nneighbor = float(self.ui.lineEdit_Nneighbor.text())
        Nneighbor = int(self.Nneighbor)

        tree = KDTree(self.gcms)
        distances, indexes = tree.query(self.gcms, Nneighbor+1) 
        self.distances = distances[:,1:] # exclude distance to the same molecule; distances has N rows (#clusters) and M columns (# neighbors)
        
        print(len(self.distances))
        indexes = indexes[:,1:]    
                
        histzWidget3 = pg.GraphicsLayoutWidget()
        histabcm = histzWidget3.addPlot(title="distances Histogram")
        
        if self.lmin != None:
            
            self.distances = self.distances[(self.distances>self.lmin) & (self.distances<self.lmax)]
            
        else:
            pass
        
        if self.lmax != None:
            
            self.distances = self.distances[(self.distances>self.lmin) & (self.distances<self.lmax)]
            
        else:
            pass
        
        if self.bins != None:
            
            bins = self.bins
        else:
            bins = 20
        
        
        histcmdist, bin_edgescmdist = np.histogram(self.distances, bins)
        widthcmdist = np.mean(np.diff(bin_edgescmdist))
        bincenterscmdist = np.mean(np.vstack([bin_edgescmdist[0:-1],bin_edgescmdist[1:]]), axis=0)
        bargraphcmdist = pg.BarGraphItem(x = bincenterscmdist, height = histcmdist, 
                                    width = widthcmdist, brush = self.brush3, pen = None)
        histabcm.addItem(bargraphcmdist)
        histabcm.setXRange(self.lmin, self.lmax)
                
        self.empty_layout(self.ui.zhistlayout_cmdist)
        self.ui.zhistlayout_cmdist.addWidget(histzWidget3)
    
        
        
    def dist_cmDBSCAN(self):
        
       
        
        scatterWidgetDBSCAN_cmdist = pg.GraphicsLayoutWidget()
        plotdistcmd = scatterWidgetDBSCAN_cmdist.addPlot(title="Clusters centers and distances")
        plotdistcmd.setAspectLocked(True)

        self.selectedcluscmd = pg.ScatterPlotItem(self.cms[:,0], self.cms[:,1], size=10, brush = self.brush3)  
        plotdistcmd.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotdistcmd.setXRange(np.min(self.xroi), np.max(self.xroi), padding=0)
        self.selectedcluscmd.sigClicked.connect(self.rx)
        
        plotdistcmd.addItem(self.selectedcluscmd)
        
        self.empty_layout(self.ui.scatterlayout_histcmdist)
        self.ui.scatterlayout_histcmdist.addWidget(scatterWidgetDBSCAN_cmdist) 
        
        

    
        
    def empty_layout(self, layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)
    
if __name__ == '__main__':
    
    app = QtWidgets.QApplication([])
    # app = QtGui.QApplication([])
    win = MPS_explorer()
    win.show()
    app.exec_()
    

    

