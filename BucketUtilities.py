# coding: utf8
"""
The following code contains all the routines used in the analysis of the 2D bucket lists generated by Spike.
The LM version contains improvements for great and faster homonuclear and heteronuclear spectra displays
L Margueritte & M-A Delsuc,  
"""
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

NETMODE = 'mieux' # standard / mieux / encore

def loadInt2D(epath, net=False, sym=False):
    """loads intensities from a csv bucket-list file from 2D spectra
    net: determines whether the cleaning method is used
        the method used is defined by NETMODE global
    sym: whether symetrisation is used
    """
    # read the file
    ne1 = pd.read_csv( epath, header=1, sep = ', ', usecols=[0, 1, 2], engine='python')
    x1 = np.array(ne1['centerF1'])
    xu1 = np.unique(x1)
    y1 = np.array(ne1['centerF2'])
    yu1 = np.unique(y1)
    z1 = np.array(ne1['bucket'])
    # matrix calculation
    Xr1, Yr1 = np.meshgrid(yu1, xu1)
    Zr1 = np.reshape(z1,(len(xu1),len(yu1)))
    netmode = NETMODE
    if net:
        if netmode=='standard':
            Zr1 = nettoie(Zr1)
        elif netmode=='mieux':
            Zr1 = nettoie_mieux(Zr1)
        elif netmode=='encore':
            Zr1 = nettoie_encore_mieux(Zr1)
        else:
            raise Exception(netmode + ' : Wrong netmode !')
    if sym:
            Zr1 = symetrise(Zr1)
    return [Xr1, Yr1, np.nan_to_num(Zr1)]
    
def loadStd2D(epath, net=False, sym=False):
    """loads std from a csv bucket-list file from 2D spectra
    net: determines whether the cleaning method is used
        the method used is defined by NETMODE global
    sym: whether symetrisation is used
    """
    # lit le fichier
    ne1 = pd.read_csv( epath, header=1, sep = ', ', usecols=[0, 1, 5], engine='python')
    x1 = np.array(ne1['centerF1'])
    xu1 = np.unique(x1)
    y1 = np.array(ne1['centerF2'])
    yu1 = np.unique(y1)
    z1 = np.array(ne1['std'])
    # calcul la matrice
    Xr1, Yr1 = np.meshgrid(yu1, xu1)
    Zr1 = np.reshape(z1,(len(xu1),len(yu1)))
    netmode = NETMODE
    if net:
        if netmode=='standard':
            Zr1 = nettoie(Zr1)
        elif netmode=='mieux':
            Zr1 = nettoie_mieux(Zr1)
        elif netmode=='encore':
            Zr1 = nettoie_encore_mieux(Zr1)
        else:
            raise Exception(netmode + ' : Wrong netmode !')
    if sym:
        Zr1 = symetrise(Zr1)
    return [Xr1, Yr1, np.nan_to_num(Zr1)]

def loadPP2D(epath, net=False, sym=False):
    """loads std from a csv bucket-list file from 2D spectra
    net: determines whether the cleaning method is used
        the method used is defined by NETMODE global
    sym: whether symetrisation is used
    """
    # lit le fichier
    ne1 = pd.read_csv( epath, header=1, sep = ', ', usecols=[0, 1, 6], engine='python')
    x1 = np.array(ne1['centerF1'])
    xu1 = np.unique(x1)
    y1 = np.array(ne1['centerF2'])
    yu1 = np.unique(y1)
    z1 = np.array(ne1['peaks_nb'])
    # calcul la matrice
    Xr1, Yr1 = np.meshgrid(yu1, xu1)
    Zr1 = np.reshape(z1,(len(xu1),len(yu1)))
    netmode = NETMODE
    if net:
        if netmode=='standard':
            Zr1 = nettoie(Zr1)
        elif netmode=='mieux':
            Zr1 = nettoie_mieux(Zr1)
        elif netmode=='encore':
            Zr1 = nettoie_encore_mieux(Zr1)
        else:
            raise Exception(netmode + ' : Wrong netmode !')
    if sym:
        Zr1 = symetrise(Zr1)
    return [Xr1, Yr1, np.nan_to_num(Zr1)]

def loadInt1D (epath) :
    """loads std from a csv bucket-list file from 1D spectra"""
    ne1 = pd.read_csv(epath, header=1, sep = ', ', usecols=[0, 1], engine='python')
    X = np.array(ne1['center'])
    Y = np.array(ne1['bucket'])
    return [X, Y]

def loadStd1D (epath) :
    """loads std from a csv bucket-list file from 1D spectra"""
    ne1 = pd.read_csv(epath, header=1, sep = ', ', usecols=[0, 4], engine='python')
    X = np.array(ne1['center'])
    Y = np.array(ne1['std'])
    return [X, Y]

def get_contour_data(ax):
    """
    Get informations about contours created by matplotlib.
    ax is the input matplotlob contour ax (cf. fig,ax produced by matplotlib)
    xs and ys are the different contour lines got out of the matplotlib. col is the color corresponding to the lines.
    """
    xs = []
    ys = []
    col = []
    isolevelid = 0
    for isolevel in ax.collections:
        isocol = isolevel.get_color()[0]
        thecol = 3 * [None]
        theiso = str(ax.collections[isolevelid].get_array())
        isolevelid += 1
        for i in range(3):
            thecol[i] = int(255 * isocol[i])
        thecol = '#%02x%02x%02x' % (thecol[0], thecol[1], thecol[2])
        for path in isolevel.get_paths():
            v = path.vertices
            x = v[:, 0]
            y = v[:, 1]
            xs.append(x.tolist())
            ys.append(y.tolist())
            col.append(thecol)
    return xs, ys, col

def affiche1D(X, Y, scale=1.0) :
    """draw the 1D bucket list of homonuclear experiment"""
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(X,Y, c='b')
    ax1.invert_xaxis()
    ax1.set_xlabel(r"ppm")
    ax1.set_ylabel(r"au")
    major_ticks = np.arange(0, 10, 0.5)
    minor_ticks = np.arange(0, 9.5, 0.25)
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor =True)
    return ax1

def affiche(X, Y, Z, scale=1.0, new=True, cmap=None, reverse=True, figsize=(10, 8), draw=False,levelbase=[0.5,1,2,5,10,20,50,100]) :
    """draw the 2D bucket list of homonuclear experiment
    To display two spectra in the same picture, use the parameters : new
    cmap : colormap of matplotlib, example : cm.winter or cm.spring"""
    if new:
        f1, ax = plt.subplots()
    else:
        ax = plt.gca()
#   else should be a drawable matplotlib axis.
    m1 = Z.max()
    level = [m1*(i/100.0)/scale for i in levelbase ]
    ax.contour(Y, X, Z, level, cmap=cmap)
    if draw:
        ax.set_xlabel(r"$\delta  (ppm)$")
        ax.set_ylabel(r"$\delta  (ppm)$")
        ax.yaxis.set_label_position('right')
        major_ticks = np.arange(0, 9.5, 0.5)
        minor_ticks = np.arange(0, 9.5, 0.1)
        ax.yaxis.set_ticks_position('right')
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor =True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
    if reverse:
        ax.invert_xaxis()
        ax.invert_yaxis()
    return ax

def affiche2(X, Y, Z, scale=1.0, new=True, cmap=None, reverse=True,draw=False,levelbase=[0.5,1,2,5,10,20,50,100]):
    """draw the 2D bucket list of heteronuclear experiment"""
    if new:
        f1, ax1 = plt.subplots()
    else:
        ax1 = plt.gca()
    m1 = Z.max()
    level = [m1*(i/100.0)/scale for i in levelbase ]
    ax1.contour(Y, X, Z, level, cmap= cmap)
    if draw:
        ax1.set_xlabel(r"$\delta  (ppm)$")
        ax1.set_ylabel(r'$\delta  (ppm)$')
        major_ticksx = np.arange(0, 10, 0.5)
        minor_ticksx = np.arange(0, 10, 0.1)
        major_ticksy = np.arange(-15, 140, 10)
        minor_ticksy = np.arange(-15, 140, 2)
        ax1.set_xticks(major_ticksx)
        ax1.set_xticks(minor_ticksx, minor = True)
        ax1.set_yticks(major_ticksy)
        ax1.set_yticks(minor_ticksy, minor = True)
    if reverse:
        ax1.invert_xaxis()
        ax1.invert_yaxis()
    return ax1

def affichegrid(X, Y, Z, scale=1.0, new=True, cmap=None, reverse=True):
    """draw the 2D bucket list of homonuclear experiment and a grid (if you need ;-) )"""
    if new:
        f1, ax1 = plt.subplots(figsize=(10, 8))
    else:
        ax1 = plt.gca()
    levelbase = [0.5,1,2,5,10,20,50,100]
    m1 = Z.max()
    level = [m1*(i/100.0)/scale for i in levelbase ]
    ax1.contour(Y, X, Z, level, cmap= cmap)
    ax1.set_xlabel(r"$\delta  (ppm)$")
    ax1.set_ylabel(r'$\delta  (ppm)$')
    major_ticks = np.arange(0, 10, 0.5)
    minor_ticks = np.arange(0, 10, 0.1)
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor = True)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor = True)
    #ax1.grid(b = True, which = 'major', axis = 'both')
    #ax1.grid(b = True, which = 'minor', axis = 'both')
    #ax1.set_xlim(xmax=11,xmin=0)
    #ax1.set_ylim(ymax=11,ymin=0)
    ax1.grid(which='major', alpha=1)
    ax1.grid(which='minor', alpha=0.3)
    if reverse:
        ax1.invert_xaxis()
        ax1.invert_yaxis()
    return ax1
    
def affratio(I1, I2, new=True, scale=1.0, cmap=None, reverse=True):
    "display the ratio of two pictures from homonuclear experiments"
    affiche(I1[0], I1[1], I1[2]/(I2[2]+1e-5), scale=scale, new=new, cmap=cmap, reverse=reverse)
    
def affratio2(I1, I2, new=True, scale=1.0, cmap=None, reverse=True):
    "display the ratio of two pictures from heteronuclear experiments"
    # il faudrait tester que les X Y sont les même !!!
    affiche2(I1[0], I1[1], I1[2]/(I2[2]+1e-5), scale=scale, new=new, cmap=cmap, reverse=reverse)
    
def affsub(I1, I2, new=True, scale=1.0, cmap=None, reverse=True):
    "display the substraction of two pictures from homonuclear experiments"
    # il faudrait tester que les X Y sont les même !!!
    affiche(I1[0], I1[1], I1[2]-I2[2], scale=scale, new=new, cmap=cmap, reverse=reverse)
    
def affsub2(I1, I2, new=True, scale=1.0, cmap=None, reverse=True):
    "display the substraction of two pictures from homonuclear experiments"
    # il faudrait tester que les X Y sont les même !!!
    affiche2(I1[0], I1[1], I1[2]-I2[2], scale=scale, new=new, cmap=cmap, reverse=reverse)    
    
def symetrise(ZZ):
    "symetrisation os spectra - simple minimum operation"
    return np.minimum(ZZ, ZZ.T)

def nettoie(ZZ, factor=2.0):
    " clean noise in matrix - hard thresholding"
    ZZr = ZZ.copy()
    thresh = factor*np.median(ZZ)
    print( thresh)
    ZZr[ZZ<thresh] = 1.0   # 1.0 allows to display log(Z) !
    return ZZr

def nettoie_mieux(ZZ, factor=2.0):
    " clean noise in matrix - hard thresholding columnwise"
    ZZr = ZZ.copy()
    for i in range(ZZ.shape[1]):
        iZZ = ZZ[:,i]
        thresh = factor*np.median(iZZ)
        ZZr[iZZ<thresh,i] = 1.0
    return ZZr

def nettoie_encore_mieux(ZZ, factor=2.0):
    " clean noise in matrix - soft thresholding columnwise"
    ZZr = ZZ.copy()
    for i in range(ZZ.shape[1]):
        iZZ = ZZ[:,i]
        thresh = factor*np.median(iZZ)
        ZZr[:,i] = np.where(iZZ<thresh, 1, iZZ-thresh+1) 
    return ZZr
    
def normalize(Z):
    "normalise les histogrammes"
    ZZ = np.log(Z)
    mu = ZZ.mean()
    sigma = ZZ.std()
    
