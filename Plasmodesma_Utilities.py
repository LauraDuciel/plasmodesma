"""
This program contains all useful functions and classes used to create the bokeh app for plasmodesma results display.

"""
import os
import yaml
import sklearn
from sklearn import linear_model
from sklearn.feature_selection import RFE, RFECV
from sklearn import preprocessing
from sklearn import utils
import numpy as np

from bokeh.layouts import column,row, widgetbox
from bokeh.models import ColumnDataSource, CustomJS,Range1d
from bokeh.models.widgets import Slider, TextInput, CheckboxGroup, Panel, Tabs
from bokeh.plotting import figure,reset_output
from bokeh.themes import Theme

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import BucketUtilities
from BucketUtilities import *

def prepare_analysis(results_folder,ref_name, Y, extension="2D/dipsi2phpr_20_bucketlist.csv",inhib=False):
    """
    prepare the arrays for rfe and linear regression.
    - results_folder: location of the plasmodesma results.
    - extension correspond to the end part of the file location. (by default, set to TOCSY extension)
    - inhib: should be True if inhibition percentage are used for Y instead of activity percentages.
    - Y is the array of activities corresponding to data present in results_folder.
    returns X and Y, arrays used to perform RFE or Linear Regression from the chosen data.
    """
    X = []
    datas = []
    for n in sorted(next(os.walk(results_folder))[1]):
        tit = os.path.join(results_folder,str(n), extension)
        if not n in ref_name: #Used to avoid importing indicated reference data. 
            datas.append(tit)
            Int = loadInt2D(tit, net=True, sym=True)
            tInt = Int[2].ravel() # [tmask]
            X.append(tInt)
    X = np.array(X)
    t = np.linspace(0, 1, 100)
    if inhib:
        def ty(x, A=0.1):
            "A defines the slope of the sigmoid"
            val = 0.5 + A*np.arctanh(2*x-1)
            val = np.minimum(val,1.0)
            val = np.maximum(val,0.0)
            return val
        Y = ty(Y)

    return X,Y

def HTproj(x, k):
    """
    returns the Hard Thresholding of x, on the ball of radius \ell_o = k
    i.e. the k largest values are kept, all other are set to 0
    """
    N = len(x)-k
    tx = np.argpartition(x, N)
    hpx = np.zeros_like(x)
    hpx[tx[N:]] = x[tx[N:]]
    return hpx

def LinRegression(X, Y, Im1, Im2, nfeatures=100):
    """
    Performs linear regression analysis between name and name2 datasets.
    - X,Y are the results of prepare_analysis.
    - name and name2 are the name of data to be displayed on which analyse is performed.
    return a mplt axis object(contour_plot) which is then passed to get_contour_data to be displayed with bokeh.
    """
    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    m = HTproj(reg.coef_, nfeatures)
    m = m.reshape(Im2[2].shape)
    return (Im1[0],Im1[1],m)

def RecurFeatElim(X,Y,Im1, Im2, nfeatures=100, CrossVal = False):
    """
    Performs RFE analysis from scikit learn between name and name2 datasets.
    - X,Y are the results of prepare_analysis.
    - name and name2 are the name of data to be displayed on which analyse is performed.
    return a mplt axis object(contour_plot) which is then passed to get_contour_data to be displayed with bokeh.
    """
    estimator = linear_model.LinearRegression()
    if CrossVal:
        selector = RFECV(estimator, step=0.5, min_features_to_select=nfeatures,cv=2)
    else:
        selector = RFE(estimator, step=0.5, n_features_to_select=nfeatures)
    selector = selector.fit(X, Y)
    N = len(Im2[2].ravel())
    m = np.zeros(N)
    m[selector.support_] = 1.0
    m = m.reshape(Im2[2].shape)
    return (Im1[0],Im1[1],m)

def LogisticRegr(X, Y, Im1, Im2, nfeatures=100):
    reg = linear_model.LogisticRegression()
    Y.astype(int)
    reg.fit(X,Y)
    m = HTproj(reg.coef_, nfeatures)
    m = m.reshape(Im2[2].shape)
    return (Im1[0],Im1[1],m)

def default_plot_settings(manip_mode):
    """
    This function is used to create a default setup for creating NMR bokeh plot.
    """
    TOOLS = "pan, box_zoom, undo, redo, reset, save"
    dbk = {'tools': TOOLS}
    dbk['x_axis_label'] = u"δ (ppm)"
    dbk['y_axis_label'] = u"δ (ppm)"
    dbk['x_range'] = Range1d(10, 0)
    if manip_mode in ("TOCSY","COSY"):
        dbk['y_range'] = Range1d(10, 0)
    elif manip_mode == "HSQC":
        dbk['y_range'] = Range1d(150, 0)
    else:
        print("No compatible mode chosen!")
    return dbk

class BokehApp_Slider_Plot(object):
    """
    Class used to create plot and associated slider from loadStd2D() results.
    """
    def __init__(self, A, B, C, manip_mode='TOCSY', dbk=None, cmap=None, title="my name", levels = [0.5,1,2,5,10,20,50,100]):
        self.A = A
        self.B = B
        self.C = C
        self.manip_mode = manip_mode
        if dbk:
            self.dbk = dbk
        else:
            self.dbk = default_plot_settings(self.manip_mode)
        self.colormap = cmap
        self.slider_value = 3.0
        self.slider_start=0.05 
        self.slider_end=30.0
        self.slider_step=0.05
        self.name =title
        self.levels = levels  
        xs,ys,col = get_contour_data(affiche(self.A, self.B, self.C, cmap=self.colormap,levelbase=self.levels,
                                             scale=self.slider_value))
        self.source = ColumnDataSource(data=dict(xs=xs, ys=ys, color=col))
        self.plot = figure(**self.dbk, title=self.name )
        self.plot.multi_line(xs='xs', ys='ys', color='color', source=self.source)
        self.scale_slider = self.slider(title=self.name, value=self.slider_value, start=self.slider_start, end=self.slider_end, 
                      step=self.slider_step)
        self.scale_slider.on_change('value', self.update_data)
        self.widget = column(self.scale_slider,self.plot)
            
    def update_data(self,attr, old, new):
        # Get the current slider value & update source
        xs,ys,col = get_contour_data( affiche( self.A, self.B, self.C, scale=self.scale_slider.value, cmap=self.colormap,levelbase=self.levels))
        self.source.data = dict(xs=xs, ys=ys, color=col)#update source
        
    def slider(self,title, value, start, end, 
                      step):
        return Slider(title=title, value=value, start=start, end=end, 
                      step=step)
    
    def add_multiline(self,E,F,G,manip_mode,title,cmap,levels):
        """
        add a multiline to the original plot and the associated slider.
        """
        new_data = BokehApp_Slider_Plot(E,F,G,dbk=self.dbk,manip_mode=manip_mode,cmap=cmap,title=title,levels=levels)
        self.plot.multi_line(xs='xs', ys='ys', color='color', source=new_data.source)
        self.widget = column(new_data.scale_slider, self.scale_slider, self.plot)


class AnalysisPlots(object):
    """
    Class used to create Analysis plots, based on the BokehApp_Slider_Plot class. Needed data are different and the slider is not for scale but features.
    """
    def __init__(self,X,Y,D1,D2,nfeatures, manip_mode='TOCSY', CrossVal=False,dbk=None, cmap=None, title="my name", levels = [0.5,1,2,5,10,20,50,100],mode="RFE"):
        self.X = X
        self.Y = Y
        self.D1 = D1
        self.D2 = D2
        self.CrossVal = CrossVal
        self.manip_mode = manip_mode
        if dbk:
            self.dbk = dbk
        else:
            self.dbk = default_plot_settings(self.manip_mode)
        self.mode=mode
        self.colormap = cmap
        self.name = title 
        self.slider_value = nfeatures
        self.slider_start = 0
        self.slider_end = 500
        self.slider_step = 5
        self.levels = levels  
        if self.mode == "RFE": 
            xs,ys,col = get_contour_data(affiche(*(RecurFeatElim(self.X, self.Y, self.D1, self.D2, CrossVal=self.CrossVal, nfeatures=self.slider_value)), cmap=self.colormap,levelbase=self.levels))
            self.source = ColumnDataSource(data=dict(xs=xs, ys=ys, color=col))
        elif self.mode == "LinReg":
            xs,ys,col = get_contour_data(affiche(*(LinRegression(self.X, self.Y, self.D1, self.D2, nfeatures=self.slider_value)), cmap=self.colormap,levelbase=self.levels))
            self.source = ColumnDataSource(data=dict(xs=xs, ys=ys, color=col))
        elif self.mode == "LogisticRegr":
            xs,ys,col = get_contour_data(affiche(*(LogisticRegr(self.X, self.Y, self.D1, self.D2, nfeatures=self.slider_value)), cmap=self.colormap,levelbase=self.levels))
            self.source = ColumnDataSource(data=dict(xs=xs, ys=ys, color=col))
        else:
            print("The chosen mode does not exist.")
        self.plot = figure(**self.dbk, title=self.name )
        self.plot.multi_line(xs='xs', ys='ys', color='color', source=self.source)
        self.nfeatures_slider = self.slider(title="N Features Kept", value=self.slider_value, start=self.slider_start, end=self.slider_end, step=self.slider_step)
        self.nfeatures_slider.on_change('value', self.update_data)
        self.widget = column(self.nfeatures_slider,self.plot)

    def update_data(self,attr, old, new):
        # Get the current slider value & update source
        if self.mode == "RFE": 
            xs,ys,col = get_contour_data(affiche(*(RecurFeatElim(self.X, self.Y, self.D1, self.D2,CrossVal=self.CrossVal, nfeatures=self.nfeatures_slider.value)), cmap=self.colormap,levelbase=self.levels))
            self.source.data = dict(xs=xs, ys=ys, color=col)
        elif self.mode == "LinReg":
            xs,ys,col = get_contour_data(affiche(*(LinRegression(self.X, self.Y, self.D1, self.D2, nfeatures=self.nfeatures_slider.value)), cmap=self.colormap,levelbase=self.levels))
            self.source.data = dict(xs=xs, ys=ys, color=col)
        elif self.mode == "LogisticRegr":
            xs,ys,col = get_contour_data(affiche(*(LogisticRegr(self.X, self.Y, self.D1, self.D2, nfeatures=self.nfeatures_slider.value)), cmap=self.colormap,levelbase=self.levels))
            self.source.data = dict(xs=xs, ys=ys, color=col)
        else:
            print("Wrong mode")

    def slider(self,title, value, start, end, step):
        return Slider(title=title, value=value, start=start, end=end, step=step)

    def add_multiline(self,E,F,G,manip_mode,title,cmap,levels):
        """
        add a multiline to the original plot and the associated scale slider.
        """
        new_data = BokehApp_Slider_Plot(E,F,G,dbk=self.dbk,manip_mode=manip_mode,cmap=cmap,title=title,levels=levels)
        self.plot.multi_line(xs='xs', ys='ys', color='color', source=new_data.source)
        self.widget = column(new_data.scale_slider, self.nfeatures_slider, self.plot)

def new_create_app(doc,folder,dataref,data_name,data_name2,netmode,activities,manip_mode,extension="2D/dipsi2phpr_20_bucketlist.csv",nfeatures=100,threshold=1E-13):
    """
    This function creates the complete bokeh application for Plasmodesma results analysis.
    - doc is the current document in which the app is made.
    - folder is the folder in which the results of plasmodesma are stored.
    - activities: the activites corresponding to the different fractions present in the folder.
    - name is the name of the first data to display & analyse.
    - name2 is the name of the second data to display & analyse.
    - netmode is the cleaning/denoising mode desired for plasmodesma (BucketUtilities).
    """
    BucketUtilities.NETMODE = netmode
    name = os.path.join(folder,data_name)
    name2 = os.path.join(folder,data_name2)
    refname = os.path.join(folder,dataref)
    Im1 = BucketUtilities.loadStd2D(name, net=True, sym=True)
    Im2 = BucketUtilities.loadStd2D(name2, net=True, sym=True)
    Imref = BucketUtilities.loadStd2D(refname, net=True, sym=True)
    X,Y =  prepare_analysis(folder,dataref, extension=extension,Y=activities)
    
    Graph1 =  BokehApp_Slider_Plot(*Im1, manip_mode=manip_mode,title=data_name)
    Graph2 =  BokehApp_Slider_Plot(*Im2, dbk=Graph1.dbk, manip_mode=manip_mode,title=data_name2)
    GraphRatio =  BokehApp_Slider_Plot(Im1[0], Im1[1], Im1[2]/(Im2[2]+1e-5),dbk=Graph1.dbk,manip_mode=manip_mode, 
                                         title="Ratio")
    GraphRatio.add_multiline(*Imref,manip_mode=manip_mode,title="Reference",cmap=cm.autumn, levels=[1])
    GraphSubstract =  BokehApp_Slider_Plot(Im1[0], Im1[1], Im1[2]-Im2[2],dbk=Graph1.dbk,manip_mode=manip_mode, 
                                             title="Subtraction")
    GraphSubstract.add_multiline(*Imref,manip_mode=manip_mode,title="Reference",cmap=cm.autumn, levels=[1])
    
    GraphRFE =  AnalysisPlots(X=X,Y=Y,D1=Im1, D2=Im2, nfeatures=nfeatures,manip_mode=manip_mode, mode="RFE",dbk=Graph1.dbk, title="RFE")
    GraphRFE.add_multiline(*Imref,manip_mode=manip_mode,title="Reference",cmap=cm.autumn, levels=[1])

    GraphRFECV =  AnalysisPlots(X=X,Y=Y,D1=Im1, D2=Im2, CrossVal=True,nfeatures=nfeatures,manip_mode=manip_mode, mode="RFE",dbk=Graph1.dbk, title="RFECV")
    GraphRFECV.add_multiline(*Imref,manip_mode=manip_mode,title="Reference",cmap=cm.autumn, levels=[1])
    
    GraphLinReg =  AnalysisPlots(X=X,Y=Y,D1=Im1, D2=Im2, nfeatures=nfeatures,manip_mode=manip_mode, mode="LinReg",dbk=Graph1.dbk, title="Linear Regression")
    GraphLinReg.add_multiline(*Imref,manip_mode=manip_mode,title="Reference",cmap=cm.autumn, levels=[1])

    # GraphLogistReg =  AnalysisPlots(X=X,Y=Y,D1=Im1, D2=Im2, nfeatures=nfeatures,manip_mode=manip_mode, mode="LogisticRegr",dbk=Graph1.dbk, title="Logisitic Regression")
    # GraphLogistReg.add_multiline(*Imref,manip_mode=manip_mode,title="Reference",cmap=cm.autumn, levels=[1])
    
    # Set up layouts and add to document
    tab1 = Panel(child=column(row(Graph1.widget,Graph2.widget),row(GraphRatio.widget,GraphSubstract.widget)), 
                 title="Visualization")
    tab2 = Panel(child=column(row(GraphRFE.widget,GraphLinReg.widget),row(GraphRFECV.widget)), title="Global Analysis")
    
    doc.add_root(Tabs(tabs=[ tab1,tab2]))
    
    doc.title = "SMARTE"

    doc.theme = Theme(json=yaml.load("""
        attrs:
            Figure:
                background_fill_color: "#DDDDDD"
                outline_line_color: white
                toolbar_location: right
                height: 470
                width: 470
            Grid:
                grid_line_dash: [6, 4]
                grid_line_color: white
    """))
    
