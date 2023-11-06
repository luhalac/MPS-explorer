# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 19:27:26 2016

@author: Federico Barabas
"""

import os
from tkinter import Tk, filedialog


def getFilename(title, types, initialdir=None):
    root = Tk()
    root.withdraw()
#    filename = filedialog.askopenfilename(title=title, filetypes=types,
#                                          initialdir=initialdir)
    filename = filedialog.askopenfilename(title=title, initialdir=initialdir)
    root.destroy()
    return filename


def getFilenames(title, types=[], initialdir=None):
    root = Tk()
    root.withdraw()
#    filenames = filedialog.askopenfilenames(title=title, filetypes=types,
#                                            initialdir=initialdir)
    filenames = filedialog.askopenfilenames(title=title, initialdir=initialdir)
    root.destroy()
    return root.tk.splitlist(filenames)


def insertSuffix(filename, suffix, newExt=None):
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt


def insertFolder(p, folder):
    splitted = os.path.split(p)
    return os.path.join(splitted[0], folder, splitted[1])
