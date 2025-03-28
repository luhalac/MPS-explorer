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
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow, QApplication
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
        self.ui.pushButton_browsefile.clicked.connect(lambda:self.select_file(1))
        self.ui.pushButton_browsefile_2.clicked.connect(lambda:self.select_file(2))
        self.ui.pushButton_scatter.clicked.connect(self.scatterplot)
        self.ui.pushButton_zrange.clicked.connect(self.update_ROI)
        self.ui.pushButton_savexyzROI.clicked.connect(lambda: self.savexyzROI(1))
        self.ui.pushButton_savexyzROI_2.clicked.connect(lambda: self.savexyzROI(2))
        self.ui.pushButton_savedistdata.clicked.connect(self.savedistdata)
        self.ui.pushButton_clusterch1.clicked.connect(lambda:self.cluster(1))
        self.ui.pushButton_clusterch2.clicked.connect(lambda:self.cluster(2))
        self.ui.pushButton_remove_bad_cluster.clicked.connect(self.dist_cm_good_clus)
        self.ui.pushButton_savecluscenters.clicked.connect(self.save_clus_CM)
        self.ui.pushButton_Distances.clicked.connect(self.KNdist_hist)
        self.ui.pushButton_saveAllClusterData.clicked.connect(lambda: self.save_dual_files(1))
        
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
        
        
        # Connect the close event to your method
        self.closeEvent = self.onCloseEvent

    def select_file(self, channel):
        try:
            root = Tk()
            root.withdraw()
            if channel == 1:
                root.filenamedata = filedialog.askopenfilename(initialdir=self.initialDir,
                                                               title='Select file')
                if root.filenamedata != '':
                    self.ui.lineEdit_filename.setText(root.filenamedata)
                    self.fileformat1 = int(self.fileformat.currentIndex())
                    self.xdata, self.ydata, self.zdata = self.import_file(root.filenamedata, self.fileformat1)
            elif channel == 2:
                root.filenamedata2 = filedialog.askopenfilename(initialdir=self.initialDir,
                                                                title='Select file')
                if root.filenamedata2 != '':
                    self.ui.lineEdit_filename_2.setText(root.filenamedata2)
                    self.fileformat2 = int(self.fileformat_2.currentIndex())
                    self.xdata2, self.ydata2, self.zdata2 = self.import_file(root.filenamedata2, self.fileformat2)
        except OSError:
            pass
        finally:
            if channel == 1 and root.filenamedata == '':
                return
            elif channel == 2 and root.filenamedata2 == '':
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
    
            self.data_points2 = np.column_stack((self.x2, self.y2))        
    
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
            diameter = 1.3*(self.circular_roi.size())
    
            # Calculate and return the radius (half of the diameter)
            radius = diameter / 2
           
            # Calculate the center coordinates
            center_x = pos.x() + size / 2
            center_y = pos.y() + size / 2
            center = np.column_stack((center_x, center_y))
            
            # Iterate through data points and check if they are inside the circular ROI
            points_inside_roi = []
            ind_inside_roi = []
            for idx,point in enumerate(self.data_points):
                if np.any(np.linalg.norm(point - center) <= radius):
                    points_inside_roi.append(point)
                    ind_inside_roi.append(idx)
            
            # Convert list of points to numpy array
            points_inside_roi = np.array(points_inside_roi)
            
            self.xroi = points_inside_roi[:,0]
            self.yroi = points_inside_roi[:,1]
                     
            
            # Define zmin and zmax
            zmin = self.ui.lineEdit_zmin.text()
            zmax = self.ui.lineEdit_zmax.text()
            
            # Convert zmin and zmax to numeric types if they are strings
            self.zmin = int(zmin) if zmin else None
            self.zmax = int(zmax) if zmax else None
            
            if self.zmax is None:
                self.zroi = self.z[ind_inside_roi]
            else:
                zroi = self.z[ind_inside_roi]
                indz = np.where((zroi > self.zmin) & (zroi < self.zmax))
                self.zroi = zroi[indz]
                self.xroi = self.xroi[indz]
                self.yroi = self.yroi[indz]
            
            
  
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
        
        else:
            
            self.xroi = self.x
            self.yroi = self.y
            
            zmin = self.ui.lineEdit_zmin.text()
            zmax = self.ui.lineEdit_zmax.text()
        
            self.zmin = int(zmin) if zmin else None
            self.zmax = int(zmax) if zmax else None
        
            if self.zmax is None:
                self.zroi = self.z
            else:
                zroi = self.z
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
        
        filename2 = self.ui.lineEdit_filename_2.text()
        
        if filename2 == '':
            
            
            pass
        
        else:
            
            if self.ui.radioButton_circROI.isChecked():
                
                # Get circular ROI position and size
                pos = self.circular_roi.pos()
                size = self.circular_roi.size()
                diameter = 1.3*(self.circular_roi.size())
        
                # Calculate and return the radius (half of the diameter)
                radius = diameter / 2
               
                # Calculate the center coordinates
                center_x = pos.x() + size / 2
                center_y = pos.y() + size / 2
                center = np.column_stack((center_x, center_y))
                
                # Iterate through data points and check if they are inside the circular ROI
                points_inside_roi2 = []
                ind_inside_roi2 = []
                for idx,point in enumerate(self.data_points2):
                    if np.any(np.linalg.norm(point - center) <= radius):
                        points_inside_roi2.append(point)
                        ind_inside_roi2.append(idx)
                
                # Convert list of points to numpy array
                points_inside_roi2 = np.array(points_inside_roi2)
                
                self.xroi2 = points_inside_roi2[:,0]
                self.yroi2 = points_inside_roi2[:,1]
                         
                
                # Define zmin and zmax
                zmin = self.ui.lineEdit_zmin.text()
                zmax = self.ui.lineEdit_zmax.text()
                
                # Convert zmin and zmax to numeric types if they are strings
                self.zmin = int(zmin) if zmin else None
                self.zmax = int(zmax) if zmax else None
                
                if self.zmax is None:
                    self.zroi2 = self.z2[ind_inside_roi2]
                else:
                    zroi = self.z2[ind_inside_roi2]
                    indz = np.where((zroi > self.zmin) & (zroi < self.zmax))
                    self.zroi2 = zroi[indz]
                    self.xroi2 = self.xroi2[indz]
                    self.yroi2 = self.yroi2[indz]
                
                
      
            elif self.ui.radioButton_squareROI.isChecked():
                
                # get square ROI position and size
                xmin, ymin = self.square_roi.pos()
                xmax, ymax = self.square_roi.pos() + self.square_roi.size()
            
                indx = np.where((self.x2 > xmin) & (self.x2 < xmax))
                indy = np.where((self.y2 > ymin) & (self.y2 < ymax))
                mask = np.in1d(indx, indy)
                ind = np.nonzero(mask)
                index = indx[0][ind[0]]
                self.xroi2 = self.x2[index]
                self.yroi2 = self.y2[index]
            
            
                zmin = self.ui.lineEdit_zmin.text()
                zmax = self.ui.lineEdit_zmax.text()
            
                self.zmin = int(zmin) if zmin else None
                self.zmax = int(zmax) if zmax else None
            
                if self.zmax is None:
                    self.zroi2 = self.z2[index]
                else:
                    zroi = self.z[index]
                    indz = np.where((zroi > self.zmin) & (zroi < self.zmax))
                    self.zroi2 = zroi[indz]
                    self.xroi2 = self.xroi[indz]
                    self.yroi2 = self.yroi[indz]
              
                    
            self.selected2 = pg.ScatterPlotItem(self.xroi2, self.yroi2, pen = self.pen2,
                                               brush = None, size = 3)  
            plotROI.setLabels(bottom=('x [nm]'), left=('y [nm]'))
            plotROI.setXRange(np.min(self.xroi2), np.max(self.xroi2), padding=0)
            plotROI.addItem(self.selected2)
            
            
            self.empty_layout(self.ui.scatterlayout_3)
            self.ui.scatterlayout_3.addWidget(scatterWidgetROI)    

            
            histz2, bin_edgesz2 = np.histogram(self.zroi2, bins='auto')
            widthzabs2 = np.mean(np.diff(bin_edgesz2))
            bincentersz2 = np.mean(np.vstack([bin_edgesz2[0:-1],bin_edgesz2[1:]]), axis=0)
            bargraphz22 = pg.BarGraphItem(x = bincentersz2, height = histz2, 
                                        width = widthzabs2, brush = self.brush2, pen = self.pen2)
            bargraphz22.setOpacity(0.5) 
            histabsz2.addItem(bargraphz22)
        
                
        self.empty_layout(self.ui.zhistlayout_2)
        self.ui.zhistlayout_2.addWidget(histzWidget2)
        
    # def savexyzROI(self, channel):
        
    #     if channel == 1:
    #         xyzROI = np.array([self.xroi, self.yroi, self.zroi])
    #         filename = self.ui.lineEdit_filename.text()
    #     elif channel == 2:
    #         xyzROI = np.array([self.xroi2, self.yroi2, self.zroi2])
    #         filename = self.ui.lineEdit_filename_2.text()
    #     else:
    #         raise ValueError("Invalid channel number")
        
    #     xyzROI = np.transpose(xyzROI)
    #     filename = os.path.splitext(filename)[0]
    #     dataNamecsv = utils.insertSuffix(filename, f'_xyzROICh{channel}.csv')
        
    #     # Export array to CSV file (using 2 decimal places)
    #     np.savetxt(dataNamecsv, xyzROI, delimiter=",", fmt="%.2f", header="x, y, z", comments="")       

     
    def savexyzROI(self, channel):
        # Suggest a default filename
        filename = self.ui.lineEdit_filename.text()
        filename = os.path.splitext(filename)[0]
        
        # Get the data based on channel
        if channel == 1:
            x_roi = self.xroi
            y_roi = self.yroi
            z_roi = self.zroi
            labels = self.dblabels if hasattr(self, 'dblabels') else None
        elif channel == 2:
            x_roi = self.xroi2
            y_roi = self.yroi2
            z_roi = self.zroi2
            labels = self.dblabels2 if hasattr(self, 'dblabels2') else None
        else:
            raise ValueError("Invalid channel number")
    
        # Create ThunderSTORM-compatible DataFrame
        data = {
            'x [nm]': x_roi,
            'y [nm]': y_roi,
            'z [nm]': z_roi,
        }
        
        # Add cluster labels if available
        if labels is not None:
            data['cluster_id'] = labels
        
        df = pd.DataFrame(data)
        
        # Open file dialog to choose save location
        file_dialog = QFileDialog()
        default_filename = utils.insertSuffix(filename, f'_xyzROICh{channel}_with_clusters.csv')
        filename, _ = file_dialog.getSaveFileName(
            caption="Save ROI Data with Clusters",
            directory=".",
            filter="CSV Files (*.csv)",
            initialFilter=default_filename
        )
        
        if filename:
            # Save with ThunderSTORM-compatible headers
            df.to_csv(filename, index=False, float_format='%.2f')
    
    # def savexyzROI(self, channel):
        
    #     # Suggest a default filename
    #     filename = self.ui.lineEdit_filename.text()
    #     filename = os.path.splitext(filename)[0]
    #     default_filename = utils.insertSuffix(filename, f'_xyzROICh{channel}.csv')
    #     # Open a file dialog to choose the location and name of the file
    #     file_dialog = QFileDialog()
    #     filename, _ = file_dialog.getSaveFileName(caption="Save ROI Data", directory=".", filter="CSV Files (*.csv)", initialFilter=default_filename)
    
    #     if filename:
    #         if channel == 1:
    #             xyzROI = np.array([self.xroi, self.yroi, self.zroi])
    #         elif channel == 2:
    #             xyzROI = np.array([self.xroi2, self.yroi2, self.zroi2])
    #         else:
    #             raise ValueError("Invalid channel number")
            
    #         xyzROI = np.transpose(xyzROI)
    #         filename = os.path.splitext(filename)[0]
    #         dataNamecsv = f'{filename}_xyzROICh{channel}.csv'
            
    #         # Export array to CSV file (using 2 decimal places)
    #         np.savetxt(dataNamecsv, xyzROI, delimiter=",", fmt="%.2f", header="x, y, z", comments="")
            
            
    
    def savedistdata(self):
        
        Nneighbor = int(self.Nneighbor)

        dist = self.distances
        
        filename = self.ui.lineEdit_filename.text()
        filename = os.path.splitext(filename)[0]
        dataNamecsv = utils.insertSuffix(filename, '_' + str(Nneighbor) + 'neighbor_distdata.csv')
        
        #export dist array to CSV file (using 2 decimal places)
        np.savetxt(dataNamecsv, dist, delimiter=",", fmt="%.2f")
        
        
    def cluster(self, channel):
        
        self.indbc = []
        
        if channel == 1:
            x_roi = self.xroi
            y_roi = self.yroi
            brushch = self.brush1
            pench = self.pen1
            scatter_layout_cluster = self.ui.scatterlayout_clusterch1
            self.minsamples = float(self.ui.lineEdit_minsamples.text())
            self.eps = float(self.ui.lineEdit_eps.text())
        elif channel == 2:
            x_roi = self.xroi2
            y_roi = self.yroi2
            brushch = self.brush2
            pench = self.pen2
            scatter_layout_cluster = self.ui.scatterlayout_clusterch2
            self.minsamples = float(self.ui.lineEdit_minsamples_2.text())
            self.eps = float(self.ui.lineEdit_eps_2.text())
        else:
            return  # Skip invalid channels
        
        # XY locs in ROI
        XY = np.column_stack((x_roi, y_roi))
        
        # DBSCAN parameters
        self.minsamples = float(self.ui.lineEdit_minsamples.text())
        min_samples = int(self.minsamples)
        
        self.eps = float(self.ui.lineEdit_eps.text())
        eps = int(self.eps)
        
        # DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(XY) 
        dblabels = db.labels_        
        
        # Store labels for saving later
        if channel == 1:
            self.dblabels = dblabels
        else:
            self.dblabels2 = dblabels
        
        # Get unique cluster labels
        unique_labels = np.unique(dblabels)
        
        # Plot clusters
        scatterWidgetcluster = pg.GraphicsLayoutWidget()
        plotclusters = scatterWidgetcluster.addPlot(title="Clustered data")
        plotclusters.setAspectLocked(True)
        plotclusters.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotclusters.setXRange(np.min(x_roi), np.max(x_roi), padding=0)
        
        for label in unique_labels:
            if label == -1:
                # Skip noise points
                continue
            
            # Get points belonging to the current cluster label
            cluster_points = XY[dblabels == label]
            
            # Plot cluster points
            cluster_plot = pg.ScatterPlotItem(cluster_points[:, 0], cluster_points[:, 1], 
                                               pen=pench, brush=pg.mkBrush(None), size=10)
            plotclusters.addItem(cluster_plot)
        
        # Plot cluster centers
        cm_list = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = XY[dblabels == label]
            cm_list.append(np.mean(cluster_points, axis=0))
        
        cms = np.array(cm_list)
        self.cms = np.around(cms, decimals=2)

        
        self.selectedcluscm = pg.ScatterPlotItem(self.cms[:, 0], self.cms[:, 1], size=10, pen=pg.mkPen('k'), brush=brushch)
        plotclusters.addItem(self.selectedcluscm)
        self.gcms = []
        self.selectedcluscm.sigClicked.connect(self.rx)
        
        # Add the plot to the UI
        self.empty_layout(scatter_layout_cluster)
        scatter_layout_cluster.addWidget(scatterWidgetcluster)

 
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
        # First save cluster centers as before
        clusCMxy = np.array([self.gcms[:,0], self.gcms[:,1]])
        clusCMxy = np.transpose(clusCMxy)
        
        filename = self.ui.lineEdit_filename.text()
        filename = os.path.splitext(filename)[0]
        
        # Save cluster centers
        centers_df = pd.DataFrame({
            'x_center [nm]': self.gcms[:,0],
            'y_center [nm]': self.gcms[:,1]
        })
        
        file_dialog = QFileDialog()
        default_centers = utils.insertSuffix(filename, '_clusCM_xy.csv')
        centers_filename, _ = file_dialog.getSaveFileName(
            caption="Save Cluster Centers",
            directory=".",
            filter="CSV Files (*.csv)",
            initialFilter=default_centers
        )
        
        if centers_filename:
            centers_df.to_csv(centers_filename, index=False, float_format='%.2f')
        
        # Optionally save additional cluster statistics
        if hasattr(self, 'dblabels'):
            # Calculate cluster sizes
            unique_labels, counts = np.unique(self.dblabels[self.dblabels != -1], 
                                           return_counts=True)
            
            stats_df = pd.DataFrame({
                'cluster_id': unique_labels,
                'size': counts,
                'x_center': [self.gcms[i,0] for i in range(len(unique_labels))],
                'y_center': [self.gcms[i,1] for i in range(len(unique_labels))]
            })
            
            default_stats = utils.insertSuffix(filename, '_cluster_stats.csv')
            stats_filename, _ = file_dialog.getSaveFileName(
                caption="Save Cluster Statistics",
                directory=".",
                filter="CSV Files (*.csv)",
                initialFilter=default_stats
            )
            
            if stats_filename:
                stats_df.to_csv(stats_filename, index=False, float_format='%.2f')
            
    def save_all_cluster_data(self, channel):
        if channel == 1:
            x_roi = self.xroi
            y_roi = self.yroi
            z_roi = self.zroi
            labels = self.dblabels if hasattr(self, 'dblabels') else None
        elif channel == 2:
            x_roi = self.xroi2
            y_roi = self.yroi2
            z_roi = self.zroi2
            labels = self.dblabels2 if hasattr(self, 'dblabels2') else None
        else:
            raise ValueError("Invalid channel number")
    
        if labels is None:
            print("No cluster data available. Run clustering first.")
            return
    
        # Create ThunderSTORM-compatible DataFrame
        df = pd.DataFrame({
            'x [nm]': x_roi,
            'y [nm]': y_roi,
            'z [nm]': z_roi,
            'frame': np.zeros(len(x_roi)),  # Required by ThunderSTORM
            'cluster_id': labels,
            'is_noise': (labels == -1).astype(int)  # 1/0 instead of True/False
        })
    
        # Add centers if available
        if hasattr(self, 'cms'):
            centers = np.full((len(labels), 2), np.nan)  # Initialize with NaNs
            valid_labels = labels[labels != -1]
            if len(valid_labels) > 0:
                unique_labels = np.unique(valid_labels)
                for i, label in enumerate(unique_labels):
                    if i < len(self.cms):  # Ensure we don't exceed available centers
                        centers[labels == label] = self.cms[i]
            df['x_center [nm]'] = centers[:, 0]
            df['y_center [nm]'] = centers[:, 1]
    
        # Ensure ThunderSTORM-required columns come first
        thunderstorm_cols = ['x [nm]', 'y [nm]', 'z [nm]', 'frame']
        other_cols = [c for c in df.columns if c not in thunderstorm_cols]
        df = df[thunderstorm_cols + other_cols]
    
        # Save with ThunderSTORM-compatible settings
        filename = QFileDialog.getSaveFileName(
            self,
            "Save all cluster data",
            os.path.splitext(self.ui.lineEdit_filename.text())[0] + f"_ch{channel}_all_cluster_data.csv",
            "CSV Files (*.csv)"
        )[0]
    
        if filename:
            df.to_csv(
                filename,
                index=False,
                sep=',',  # Explicit comma separator
                float_format='%.2f',  # 2 decimal places
                encoding='utf-8'
            )
            print(f"Saved all cluster data file: {filename}") 
    
    def save_thunderstorm_minimal(self, channel):
        """Saves a minimal x,y,z-only CSV compatible with ThunderSTORM"""
        # Get data for selected channel
        if channel == 1:
            x, y, z = self.xroi, self.yroi, self.zroi
        elif channel == 2:
            x, y, z = self.xroi2, self.yroi2, self.zroi2
        else:
            raise ValueError("Invalid channel")
    
        # Create minimal DataFrame with ThunderSTORM-required columns
        ts_df = pd.DataFrame({
            'id': np.arange(len(x)),  # Required by ThunderSTORM
            'frame': np.zeros(len(x)),  # Required by ThunderSTORM
            'x [nm]': x,
            'y [nm]': y,
            'z [nm]': z,
            'sigma [nm]': np.full(len(x), 20.0),  # Dummy values required by ThunderSTORM
            'intensity [photon]': np.full(len(x), 1000.0),
            'offset [photon]': np.zeros(len(x)),
            'bkgstd [photon]': np.full(len(x), 10.0),
            'uncertainty [nm]': np.full(len(x), 5.0)
        })
    
        # Get save path
        base_name = os.path.splitext(self.ui.lineEdit_filename.text())[0]
        ch_suffix = f"_ch{channel}" if channel else ""
        default_name = f"{base_name}{ch_suffix}_thunderstorm_minimal.csv"
        
        # Get save path from user
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Minimal ThunderSTORM CSV",
            default_name,
            "CSV Files (*.csv)"
        )
        

        
        if filename:
            # Save with ThunderSTORM-required format
            ts_df.to_csv(
                filename,
                index=False,
                sep=',',
                float_format='%.1f',  # ThunderSTORM typically uses 1 decimal
                encoding='utf-8'  # Windows line endings
            )
            print(f"Saved minimal ThunderSTORM file: {filename}")
     
    def save_dual_files(self, channel):
        """Saves both full cluster data and minimal ThunderSTORM files"""
        # First save the full cluster data
        self.save_all_cluster_data(channel)
        
        # Then save the minimal version
        self.save_thunderstorm_minimal(channel)
    
    def latchange(self):
        
        self.bins = int(self.nbins.text())
        self.lmin = float(self.latmin.text())
        self.lmax = float(self.latmax.text())
    
        
      
        self.KNdist_hist()
        
    def KNdist_hist(self):
        
        # compute distances to nearest neighbors (cm of the clusters obtained with DBSCAN)
        self.Nneighbor = float(self.ui.lineEdit_Nneighbor.text())
        Nneighbor = int(self.Nneighbor)
        print(Nneighbor)
        
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
            
            
    def onCloseEvent(self, event):
        # Stop the entire process and close the application
        QApplication.quit()
    
if __name__ == '__main__':
    
    app = QtWidgets.QApplication([])
    # app = QtGui.QApplication([])
    win = MPS_explorer()
    win.show()
    app.exec_()
    

    

