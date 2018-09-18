#!/usr/bin/env python 
# encoding: utf-8

"""
Program to compute peak and bucket list from a set of 1D and 2D

Petar Markov & M-A Delsuc,  First version

v1: renamed to Plasmodesma, made it working   PM MAD
v2: added report(), many cosmetic improvements  MAD
v3: added DOSY processing, apmin, external parameters + many improvements   MAD
v4: added multiprocessing    MAD
v5: added bk_ftF1 and adapted to python3    MAD
v6: final version for publication
"""
from __future__ import print_function, division

#---------------------------------------------------------------------------
#1. Imports; From global to specific modules in descending order
import sys
# mettre ci-dessous l'adresse de spike, ou bien rien si spike est à côté de ce programme
#sys.path.append('/Users/rdc/PALMA_WEB') # Path to the directory where SPIKE can be found


from glob import glob
import os.path as op
import os
import tempfile
import zipfile as zip
import datetime
import re
import sys #
try:
    import ConfigParser
except:
    import configparser as ConfigParser
try:
    import copy_reg
except:
    import copyreg as copy_reg
import types
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

print ("**********************************************************************************")
print ("*                              PLASMODESMA program                               *")
print ("*           - automatic advanced processing of NMR experiment series -           *")
print ("*                                                                                *")
print ("**********************************************************************************")
print ("Loading utilities ...")

import spike
from spike.Algo.BC import correctbaseline # Necessary for the baseline correction
import spike.File.BrukerNMR as bk
import spike.NPKData as npkd
from spike.NPKData import as_cpx
from spike.util.signal_tools import findnoiselevel

import Bruker_Report

#---------------------------------------------------------------------------
#2. Parameters
# These can be changed to tune the program behavior
BC_ITER = 5     # Used for baselineC orrection; It is advisable to use a larger number for iterating, e.g. 5
TMS = True      # if true, TMS (or any 0 ppm reference) is supposed to be present and used for ppm calibration
SANERANK = 20   # used for denoising of 2D experiments, typically 10-50; setting to 0 deactivates denoising
DOSY_LAZY = True    # if True, will not reprocess DOSY experiment if an already processed file is on the disk
NPROC = 2           # The number of processor fr DOSY calculation, Default value = 1;  value >1 will activate multiprocessing mode
PALMA_ITER = 20000  # used for processing of DOSY

#---------------------------------------------------------------------------
#3. Utilities
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

def mkdir(f):
    "If a folder doesn't exist it is created"
    if not op.exists(f):
        os.makedirs(f)

def FT1D(numb1, ppm_offset=0, autoph=True, ph0=0, ph1=0):
    "Performs FT and corrections of experiment 'numb1' and returns data"
    def phase_from_param():
        "read the proc parame file, and returns phase parameters ok for spike"
        #print( proc['$PHC0'], proc['$PHC1'] )
        ph1 = -float( proc['$PHC1'] ) # First-order phase correction
        ph0 = -float( proc['$PHC0'] )+ph1/2 # Zero-order phase correction
        zero = -360*d.axis1.zerotime
        #print (ph0, ph1)
        return (ph0, ph1+zero)

    proc = bk.read_param(numb1[:-3]+'pdata/1/procs')
    d = bk.Import_1D(numb1)
    d.apod_em(1,1).zf(2).ft_sim()
    if not autoph:
        p0,p1 = phase_from_param()
        d.phase( p0+ph0, p1+ph1 )   # Performs the stored phase correction
    else:
        d.bruker_corr().apmin()     # automatic phase correction
    d.unit = 'ppm'
    d.axis1.offset += ppm_offset*d.axis1.frequency

    spec = np.real( as_cpx(d.buffer) )
    # the following is a bit convoluted baseline correction, 
    bl = correctbaseline(spec, iterations=BC_ITER, nbchunks=d.size1//1000)
    dd = d.copy()
    dd.set_buffer(bl)
    dd.unit = 'ppm'
    d.real()
    d -= dd # Equal to d=d-dd; Used instead of (spec-bl)
    
    return d

def autozero(d, z1=(0.1,-0.1), z2=(0.1,-0.1),):
    """
    This function search for a peak around 0ppm, assumed to be the reference compound (TMS)
    and assign it to exactly 0
    z1 and z2 (not used in 2D) are the zoom window in which the peak is searched
    """
    # peak pick TMS
    sc = 25                     # scaling for pp threshold
    d.absmax = np.nanmax( np.abs(d.buffer) )
    d.peaks=[]          # initialize the loop
    while len(d.peaks)==0 and sc<400:
        sc *= 2.0
        if d.dim == 1:
            d.pp(zoom=z1, threshold=d.absmax/sc)   # peak-pick around zero
        elif d.dim == 2:
            d.pp(zoom=(z1,z2),  threshold=d.absmax/sc)   # peak-pick around zero
    # exit if nothing at max sc
    if len(d.peaks) == 0:
        print("**** autozero does not find the TMS peak ****")
        return d
    # then set to 0
    d.peaks.largest()       # sort largest first
    d.centroid()            # optimize the peak
    if d.dim == 1:
        d.axis1.offset -= d.axis1.itoh(d.peaks[0].pos)  # and do the correction
    elif d.dim == 2:
        d.axis2.offset -= d.axis2.itoh(d.peaks[0].posF2)
        d.axis1.offset -= d.axis1.itoh(d.peaks[0].posF1)
    del(d.peaks)
    return d
    
def get_config(base, manip, fidname):
    """
    reads the parameters.cfg file that is located in the root of the processing
    it contains parameters, one section per procno
    [manip/1]
    ppm_offset = 1.2
    ph0 = 30
    ph1 -60
    return a tuples with the values

    missing values are set to zero
    no effect if the file is absent
    """ 
    vals = [0,0,0]
    configfile = op.join(base,"parameters.cfg")
    if op.exists(configfile):
        print ('reading parameters.cfg')
        cp = ConfigParser.SafeConfigParser()
        cp.read(configfile)
        for i,p in enumerate(["ppm_offset", "ph0", "ph1"]):
            try:
                vals[i] = cp.getfloat("%s/%s"%(manip,fidname),p)
            except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
                pass
            else:
                print(p,vals[i])
    return tuple(vals)

#---------------------------------------------------------------------------
#4. Main code
def process_1D(exp, resdir):
    "Performs all processing of exp, and produces the spectrum (with and without peaks) and the list files"
    fiddir =  op.dirname(exp)
    basedir, fidname = op.split(fiddir)
    base, manip =  op.split(basedir)

    ppm_offset, ph0, ph1 = get_config(base, manip, fidname)

    d = FT1D(exp, ppm_offset=ppm_offset, ph0=ph0, ph1=ph1)
    if TMS:
        d = autozero(d)

    noise = findnoiselevel( d.get_buffer() )
    d.pp(50*noise)
    d.centroid()            # optimize the peaks
    
    d.unit = 'ppm'
    d.display(label="%s/%s"%(manip,fidname))
    plt.savefig( op.join(resdir, '1D', fidname+'.pdf') ) # Creates a PDF of the 1D spectrum without peaks
    d.display_peaks() # peaks.display(f=d.axis1.itop)
    plt.savefig( op.join(resdir, '1D', fidname+'_pp.pdf') ) # Creates a PDF of the 1D spectrum with peaks
    plt.close()

    pkout = open( op.join(resdir, '1D', fidname+'_peaklist.csv')  , 'w') 
    d.peaks.report(f=d.axis1.itop, file=pkout)
    pkout.close()

    bkout = open( op.join(resdir, '1D', fidname+'_bucketlist.csv')  , 'w')
    #d.bucket1d(file=bkout)
    d.bucket1d(file=bkout, bsize=0.01)
    bkout.close()
    d.save(op.join( fiddir,"processed.gs1") )
    return d

def process_2D(numb2, resdir):
    "Performs all processing of experiment 'numb2' and produces the spectrum with and without peaks"
    fiddir =  op.dirname(numb2)
    basedir, fidname = op.split(fiddir)
    base, manip =  op.split(basedir)
    exptype =  bk.read_param(bk.find_acqu( fiddir ) ) ['$PULPROG']
    exptype =  exptype[1:-1]  # removes the <...>

    ppm_offset, ph0, ph1 = get_config(base, manip, fidname)

    d = bk.Import_2D(numb2)
    d.unit = 'ppm'
    scale = 10.0 
    sanerank = SANERANK

    #1. If TOCSY  
    if 'dipsi' in exptype:
        print ("TOCSY")
#        print exptype
        if 'dipsi2ph' in exptype:
            d.apod_sin(maxi=0.5, axis=2).zf(zf2=2).ft_sim()
            if sanerank != 0:
                d.sane(rank=sanerank, axis=1)
            d.apod_sin(maxi=0.5, axis=1).zf(zf1=4).ft_sh_tppi().modulus().rem_ridge()
        elif 'dipsi2etgpsi' in exptype:
            d.apod_sin(maxi=0.5, axis=2).zf(zf2=2).ft_sim()
            if sanerank != 0:
                d.sane(rank=sanerank, axis=1)
            d.apod_sin(maxi=0.5, axis=1).zf(zf1=4).ft_n_p().modulus().rem_ridge()
        scale = 50.0
        d.axis2.offset += ppm_offset*d.axis2.frequency
        if TMS:
            d = autozero(d)

    #2. If COSY DQF
    elif 'cosy' in exptype:
        print ("COSY DQF")
        d.apod_sin(maxi=0.5, axis=2).zf(zf2=2).ft_sim()
        if sanerank != 0:
            d.sane(rank=sanerank, axis=1)
        d.apod_sin(maxi=0.5, axis=1).zf(zf1=4).ft_sh_tppi().modulus().rem_ridge()
        scale = 20.0
        d.axis2.offset += ppm_offset*d.axis2.frequency
        if TMS:
            d = autozero(d)
    #3. If HSQC
    elif 'hsqc' in exptype:
        print ("HSQC")
        if 'ml' in exptype:
            print ("TOCSY-HSQC")
        d.apod_sin(maxi=0.5, axis=2).zf(zf2=2).ft_sim().conv_n_p()
        if sanerank != 0:
            d.sane(rank=sanerank, axis=1)
        d.apod_sin(maxi=0.5, axis=1).zf(zf1=4).ft_sh().modulus().rem_ridge()
        scale = 10.0
        d.axis2.offset += ppm_offset*d.axis2.frequency
        if TMS:
            d = autozero(d, z1=(5,-5))

    #4. If HMBC
    elif 'hmbc' in exptype:
        print ("HMBC")
        if 'hmbc' in exptype:
            print ("HMBC")
        d.apod_sin(maxi=0.5, axis=2).zf(zf2=2).ft_sim()
        if 'et' in exptype:
            d.conv_n_p()
        if sanerank != 0:
            d.sane(rank=sanerank, axis=1)
        d.apod_sin(maxi=0.5, axis=1).zf(zf1=4).bk_ftF1().modulus().rem_ridge() # For Pharma MB1-X-X series
        scale = 10.0
        d.axis2.offset += ppm_offset*d.axis2.frequency
        if TMS:
            d = autozero(d, z1=(5,-5))

    #5. If DOSY - Processed in process_DOSY
    elif 'ste' in exptype or 'led' in exptype:
        print ("DOSY")
        d = process_DOSY(numb2, ppm_offset, lazy=DOSY_LAZY)
        scale = 50.0

    d.save(op.join(fiddir,"processed.gs2"))
    d.display(scale=scale)
    plt.savefig( op.join(resdir, '2D', exptype+'_'+fidname+'.pdf') ) # Creates a PDF of the 2D spectrum without peaks
    analyze_2D( d, name=op.join(resdir, '2D', exptype+'_'+fidname) )
    plt.savefig( op.join(resdir, '2D', exptype+'_'+fidname+'_pp.pdf') ) # Creates a PDF of the 2D spectrum with peaks
    plt.close()
    return d

def process_DOSY(fid, ppm_offset, lazy=False):
    "Performs all processing of DOSY "
    import spike.plugins.PALMA as PALMA
    global POOL
    d = PALMA.Import_DOSY(fid)
    print('PULPROG', d.params['acqu']['$PULPROG'],'   dfactor', d.axis1.dfactor)
    # process in F2
    processed = op.join( op.dirname(fid),'processed.gs2' )
    if op.exists( processed ) and lazy:
        dd = npkd.NPKData(name=processed)
        npkd.copyaxes(d, dd)
        dd.axis1.itype = 0
        dd.axis2.itype = 0
        dd.adapt_size()
    else:
        d.chsize(sz2=min(16*1024,d.axis2.size))
        d.apod_em(1,axis=2).ft_sim().bruker_corr()
        # automatic phase correction
        r = d.row(2)
        r.apmin()
        d.phase(r.axis1.P0, r.axis1.P1, axis=2).real()
        # correct
        d.axis2.offset += ppm_offset*d.axis2.frequency
        # save
        fiddir =  op.dirname(fid)
        d.save(op.join(fiddir,"preprocessed.gs2"))
        # ILT
        NN = 256
        d.prepare_palma(NN, 10.0, 10000.0)
        mppool = POOL
        dd = d.do_palma(miniSNR=20, nbiter=PALMA_ITER, lamda=0.05, mppool=mppool )
        if TMS:
            r = autozero(r)  # calibrate only F2 axis !
            dd.axis2.offset = r.axis1.offset
    dd.axis2.currentunit = 'ppm'
    return dd


def analyze_2D(d, name, pplevel=10):
    "Computes peak and bucket lists and exports them as CSV files"
    from spike.NPKData import NMRAxis
    dd = d.copy() # Removed because of error with 'sane' algorithm

    dd.sg2D(window_size=7, order=2) # small smoothing
    noise = findnoiselevel( dd.get_buffer().ravel() )
    threshold = pplevel*noise
    if noise == 0:          # this might happen on DOSY because of 0 values in empty columns
        rr = dd.get_buffer().ravel()
        threshold = pplevel*findnoiselevel( rr[rr>0] )
    dd.pp(threshold)
    try:
        d.centroid()            # optimize the peaks
    except AttributeError:
        pass
    dd.display_peaks(color="g")
    
    pkout = open( name+'_peaklist.csv'  , 'w')
    dd.report_peaks(file=pkout)
    pkout.close() 
    bkout = open( name+'_bucketlist.csv'  , 'w')
    if name.find('cosy') != -1 or name.find('dipsi') != -1:
        dd.bucket2d(file=bkout, zoom=( (0.5, 9.5) , (0.5, 9.5)), bsize=(0.03, 0.03) )
    elif name.find('hsqc') != -1 or name.find('hmbc') != -1:
        dd.bucket2d(file=bkout, zoom=( (-10,150) , (0.5, 9.5)), bsize=(1.0, 0.03) )
    elif name.find('ste') != -1 or name.find('led') != -1:
        ldmin = np.log10(d.axis1.dmin)
        ldmax = np.log10(d.axis1.dmax)
        sw = ldmax-ldmin
        dd.buffer[:,:] = dd.buffer[::-1,:]  # return axis1 
        dd.axis1 = NMRAxis(specwidth=100*sw, offset=100*ldmin, frequency = 100.0, itype = 0)     # faking a 100MHz where ppm == log(D)
        dd.bucket2d(file=bkout, zoom=( (ldmin, ldmax) , (0.5, 9.5)), bsize=(0.1, 0.03) ) #original parameters
        return ('DOSY')
    else:
        print ("*** Name not found!")
    bkout.close()

def process_sample(sample, resdir):
    "Redistributes NMR experiment to corresponding processing"
    sample_name = op.basename(sample)
    print (sample_name)
    for exp in glob( op.join(sample, "fid") ): # For 1D processing
        print (exp)
        process_1D(exp, resdir)
    for exp in glob( op.join(sample, "ser") ): # For 2D processing
        print (exp)
        process_2D(exp, resdir)

def analysis_report(resdir, fname):
    """
    Generate a csv report for all bucket lists and peak lists found during processing
    """
    with open(fname,'w') as F:
        print("# report from", resdir, file=F)                                 # csv comment
        print("manip, expno, type, file, content", file=F )
        for exp in glob(op.join(resdir,'*')):
            for f1d in glob(op.join(exp, '1D', '*.csv')):   # all 1D
                csvname = (op.basename(f1d))
                csvsplit = csvname.split('_')
                firstl = open(f1d,'r').readline()
                print (op.basename(exp), csvsplit[0], '1D', csvname, firstl[1:], sep=',', file=F)
            for f2d in glob(op.join(exp, '2D', '*.csv')):   # all 1D
                csvname = (op.basename(f2d))
                csvsplit = csvname.split('_')
                firstl = open(f2d,'r').readline()
                print (op.basename(exp), csvsplit[1], csvsplit[0], csvname, firstl[1:], sep=',', file=F)

#---------------------------------------------------------------------------
def main(DIREC):
    "Creates a new directory for every sample along with subdirectories for the 1D and 2D data"
    import traceback
    global POOL
    if NPROC > 1:
        print('Processing on %d processors'%NPROC)
        copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
        POOL = mp.Pool(NPROC)
    else:
        POOL = None

    if not op.isdir(DIREC):
        raise Exception("\n\nDirectory %s is non-valid"%DIREC)
    if len( glob( op.join(DIREC, '*') ) )==0:
        print( "WARNING\n\nDirectory %s is empty"%DIREC)
    for sp in glob( op.join(DIREC, '*') ):
        # validity of sp
        if not op.isdir(sp):
            break
        if sp == 'Results': # leftovers...
            print("Results from a previous run is present, stopping...")
            break
        # ok, continue
        resdir = op.join( DIREC, 'Results', op.basename(sp) )
        mkdir(resdir)
        for folder in ['1D', '2D']:
            mkdir( op.join(resdir, folder) )
        try:
            process_sample(sp, resdir)
        except IOError:
            print("**** ERROR with file {}\n---- not processed\n".format(sp))
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
    Bruker_Report.generate_report( DIREC, op.join(DIREC, 'report.csv') )
    analysis_report(op.join( DIREC, 'Results'), op.join( DIREC,'analysis.csv'))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', action='store', dest='DIREC', help='-DIRECTORY_with_NMR_experiments')
    args = parser.parse_args()

    print ("Processing ...")        
    main(args.DIREC)
