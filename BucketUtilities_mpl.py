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

import glob
from os import path as op
import xarray as xr
#from BucketUtilities import nettoie, nettoie_mieux, nettoie_encore_mieux, symetrise, get_contour_data, affiche

NETMODE = 'mieux' # standard / mieux / encore

class StatSeries():
    """to store a series of StatSpectrum()"""
    def __init__(self, folder, data_name, data_name2, activities, manip_mode='TOCSY', dataref=None, sym=True, net=True, normalize=False, debug=True):   # load all datasets
        """
        folder : where are the datasets (Results/)
        manip : which to load - the begin of the names (dipsi cosy hsqc ...)
        data_name : which to keep data_name2, activities
        """
        self.Series = []
        self.keys = None
        self.reference = []
        self.manip_mode = manip_mode
        self.folder = folder
        # load experiments
        if manip_mode == 'TOCSY':
            maniplist = ['dipsi', 'mlev', 'towny']
            mode = 'homonuclear'
        elif manip_mode == 'COSY':
            maniplist = ['cosy']
            mode = 'homonuclear'
        elif manip_mode == 'DOSY':
            maniplist = ['ste', 'led', '%led']
            mode = 'dosy'
        elif manip_mode == 'HSQC':
            maniplist = ['hsqc', 'hmbc']
            mode = 'heteronuclear'
        else:
            raise Exception('wrong mode, use only one of HSQC, COSY, TOCSY, DOSY')
        for n in sorted( glob.glob( op.join(folder, '*', '2D', '*_bucketlist.csv' ))):
            seqname = op.basename(n)
            if any( (seqname.startswith(manip) for manip in maniplist ) ):
                if dataref is not None and \
                    any( (iref in n for iref in dataref) ):
                        self.reference.append(StatSpectrum(n, sym=sym, net=net, normalize=normalize, manip_mode=mode))
                        if debug: print("loaded %s\n   as reference"%n)
                else:
                    self.Series.append( StatSpectrum(n, sym=sym, net=net, normalize=normalize, manip_mode=mode) )
                    if debug: print("loaded %s"%n)
                    if data_name in n:
                        self.data1 = self.Series[-1]
                        self.indexdata1 = len(self.Series)-1
                        if debug: print("   as data1")
                    if data_name2 in n:
                        self.data2 = self.Series[-1]
                        self.indexdata2 = len(self.Series)-1
                        if debug: print("   as data2")
                    if self.keys is None:
                        self.keys = self.Series[-1].keys
                    elif (self.keys != self.Series[-1].keys).any():
                        print("** WARNING, DATA Series %s is not homogeneous"%(folder))
        # set parameters
        self.activities = activities
        self.Y = activities
    def X(self, key):
        """return a preformated array, ready for scikit-learn RegLin"""
        return np.array([ d.Data.loc[key].values.ravel() for d in self.Series ])

class StatSpectrum():
    """package all information from buckets into one single object"""
    def __init__(self, epath, net=False, sym=False, manip_mode='homonuclear', normalize=False, extend=True):
        self.epath = epath
        # SMARTE_v3/Results/ARTEref_161123/2D/dipsi2phpr_20_bucketlist.csv
        self.name = epath.split(op.sep)[2]
        self.net = net
        self.sym = sym
        self.manip_mode = manip_mode
        self.Xr1 = None
        self.Yr1 = None
#        self.Zr1 = []  # usefull ?
        self.normalize = normalize
        self.extend = extend
        self.loadResult2D()
        self.F1_values = self.xu1
        self.F2_values = self.yu1

    def loadResult2D(self):
        """loads all entries from a csv bucket-list file from 2D spectra
        net: determines whether the cleaning method is used
            the method used is defined by NETMODE global
        sym: whether symetrisation is used
        """
        # read the file - build axes
        def clean(arr, nett):
            "used to normalize, clean (if nett is True), and symmetrize the array"
            if self.normalize:
                z1 = arr/arr.max()
            else:
                z1 = arr
            z1 = z1.values.reshape((len(self.xu1), len(self.yu1)))
            if nett:
                if netmode=='standard':
                    z1 = nettoie(z1)
                elif netmode=='mieux':
                    z1 = nettoie_mieux(z1)
                elif netmode=='encore':
                    z1 = nettoie_encore_mieux(z1)
                else:
                    raise Exception(netmode + ' : Wrong netmode !')
            if self.sym and self.manip_mode == 'homonuclear':
                    z1 = symetrise(z1)
            return np.nan_to_num(z1)
        # load
        ne1 = pd.read_csv( self.epath, header=1, sep = ', ', engine='python')
        x1 = np.array(ne1['centerF1'])
        self.xu1 = np.unique(x1)
        y1 = np.array(ne1['centerF2'])
        self.yu1 = np.unique(y1)
        # read data
        netmode = NETMODE
        zd_accu = {}   # used to build final xarray
        for k in ne1.keys()[2:-2]: # remove 2 first (coordinates) and 2 last (sizes)
            zd_accu[k] = clean(ne1[k], self.net)
        if self.extend:
            zd_accu["bucket_x_std"] = clean( ne1["bucket"] * ne1['std'], self.net)
            zd_accu["bucket_d_std"] = clean( ne1["bucket"] / ne1['std'], self.net)
            zd_accu["min_max"] = clean( ne1["max"] - ne1['min'], self.net)
            zd_accu["log_bucket"] = clean( np.log(abs(ne1["bucket"])), False )
            zd_accu["log_std"] = clean( np.log(ne1["std"]), False )
            zd_accu["log_min_max"] = clean( np.log(ne1["max"] - ne1['min']), False)
            try:
                zd_accu["nbpk_x_std"] = clean( ne1["peaks_nb"] * ne1['std'], self.net)
                zd_accu["nbpk_x_log_std"] = clean( ne1["peaks_nb"] *  np.log(ne1['std']), False)
            except:
                print('Skipping nbpk entries')
        # matrix calculation
        Zr1 = np.array(list(zd_accu.values())) 
        self.keys = pd.Index(zd_accu.keys())
        self.Data = xr.DataArray(Zr1,
            dims=('type','F1','F2'),
            coords={'type':self.keys, 'F1':self.xu1, 'F2':self.yu1})
    
    def myMesh(self):
        return np.meshgrid(self.yu1, self.xu1)

def affiche_contour(*arg, **kwarg):
    return get_contour_data(affiche(*arg, **kwarg))
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
    plt.close(f1)
    return ax
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