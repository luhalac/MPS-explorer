MPS exlorer. py is the code to explore and analyse MPS data


data explorer. ui is the designer file that contains GUI design


MPS Data Explorer Documentation
Overview
A PyQt5-based GUI application for exploring and analyzing multi-channel localization microscopy data (x,y,z coordinates) with clustering capabilities.

Key Features
Dual-channel data visualization and analysis

ROI selection (circular or rectangular)

Z-range filtering

DBSCAN clustering with manual cluster selection

Nearest-neighbor distance analysis

Data export in multiple formats

Class Structure
MPS_explorer(QMainWindow)
Main application window class.

Key Methods:
__init__() - Initializes UI and connects signals/slots

select_file(channel) - File selection dialog for specified channel

import_file(filename, fileformat) - Handles different input formats:

0: Picasso HDF5

1: ThunderStorm CSV

2: Custom CSV

scatterplot() - Creates 2D scatter plots with ROI tools

update_ROI() - Updates displayed data based on ROI selection

cluster(channel) - Performs DBSCAN clustering on selected channel

save_all_clustered_data(channel) - Exports clustered data with options

Data Processing
Input Handling:

Supports multiple file formats

Automatic pxsize conversion for Picasso files (133nm)

Visualization:

Interactive scatter plots with ROI tools

Z-histograms for both channels

Cluster visualization with color coding

Analysis:

DBSCAN clustering with adjustable parameters

Manual cluster selection/rejection

Nearest-neighbor distance calculations

Distance histograms with adjustable bins

UI Components
File Selection: Combo boxes for format selection + browse buttons

ROI Controls: Shape selection + Z-range filtering

Clustering Parameters: eps and min_samples inputs

Visualization Areas: Multiple plot areas for raw data and analysis results

Export Buttons: Various options for saving processed data

Usage
Load data files for each channel

Adjust ROI and z-range as needed

Perform clustering with desired parameters

Manually select/reject clusters as needed

Export results in desired format

Dependencies
PyQt5

pyqtgraph

numpy

pandas

scikit-learn (DBSCAN, KDTree)

h5py (for Picasso files)

File Structure
Main GUI code with all functionality

UI file (data_explorer.ui) designed in Qt Designer

Companion tools module (tools.utils)
