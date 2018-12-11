"""
This program contains all useful functions and classes used to create the bokeh app for plasmodesma results display.

"""
import os
import yaml
from copy import copy

from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import numpy as np

from bokeh.layouts import column,row
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import Slider, Panel, Tabs
from bokeh.plotting import figure
from bokeh.themes import Theme

import matplotlib.cm as cm

import BucketUtilities_mpl as BU

def ty(x, A, B):
    "A defines the slope of the sigmoid"
    if A+B == 0 :
        return x
    val = B + A*np.arctanh(2*x-1)
    val = np.minimum(val,1.0)
    val = np.maximum(val,0.0)
    return val

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

def LinRegression(X, Y, D1, D2, nfeatures=100):
    """
    Performs linear regression analysis between name and name2 datasets.
    - X,Y are the results of prepare_analysis.
    - name and name2 are the name of data to be displayed on which analyse is performed.
    return a mplt axis object(contour_plot) which is then passed to BU.affiche_contour to be displayed with bokeh.
    """
    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    m = HTproj(reg.coef_, nfeatures)
    m = m.reshape((len(D1),len(D2)))
    return (D1, D2, m)

def ridge(X,Y,D1,D2,nfeatures=100):
    """
    Performs ridge regression
    """
    ridge = linear_model.Ridge(alpha = 7)
    ridge.fit(X,Y)
    m = HTproj(ridge.coef_, nfeatures)
    m = m.reshape((len(D1),len(D2)))
    return(D1,D2,m)

def elasticnet(X,Y,D1,D2,nfeatures=100):
    """
    performs elastic net regression
    """
    elasticnet = linear_model.ElasticNet(alpha = .01)
    elasticnet.fit(X,Y)
    m = HTproj(elasticnet.coef_, nfeatures)
    m = m.reshape((len(D1),len(D2)))
    return(D1,D2,m)

def RecurFeatElim(X, Y, D1, D2, nfeatures=100):
    """
    Performs RFE analysis from scikit learn between name and name2 datasets.
    - X,Y are the results of prepare_analysis.
    - name and name2 are the name of data to be displayed on which analyse is performed.
    return a mplt axis object(contour_plot) which is then passed to BU.affiche_contour to be displayed with bokeh.
    """
    estimator = linear_model.LinearRegression()
    selector = RFE(estimator, step=0.5, n_features_to_select=nfeatures)
    selector = selector.fit(X, Y)
    N = X.shape[1]
    m = np.zeros(N)
    m[selector.support_] = 1.0
    m = m.reshape((len(D1),len(D2)))
    return (D1, D2, m)

def LogisticRegr(X, Y, Im1, Im2, nfeatures=100):
    reg = linear_model.LogisticRegression()
    Y.astype(int)
    reg.fit(X, Y)
    m = HTproj(reg.coef_, nfeatures)
    m = m.reshape(Im2[2].shape)
    return (Im1[0], Im1[1], m)

def default_plot_settings(manip_mode):
    """
    This function is used to create a default setup for creating NMR bokeh plot.
    """
    TOOLS = "pan, box_zoom, undo, redo, reset, save"
    dbk = {'tools': TOOLS}
    dbk['x_axis_label'] = u"δ (ppm)"
    dbk['y_axis_label'] = u"δ (ppm)"
    dbk['x_range'] = Range1d(10, 0)
    dbk['plot_width'] = 420 
    dbk['plot_height'] = 420
    if manip_mode in ("TOCSY","COSY","homonuclear"):
        dbk['y_range'] = Range1d(10, 0)
    elif manip_mode in ("HSQC", "heteronuclear"):
        dbk['y_range'] = Range1d(150, 0)
    else:
        print("No compatible mode chosen!",manip_mode)
    return dbk

def slider(title, value, start, end, step):
    """
    Creates a Bokeh Slider
    """
    return Slider(title=title, value=value, start=start, end=end, 
                      step=step)

class BokehApp_Slider_Plot_mpl(object):
    """
    Class used to create plot and associated slider from StatSpectrum() results.
    """
    def __init__(self, StatSpect, display, dbk=None, cmap=None, title=None, levels=None, debug=False):
        self.StatSpect = StatSpect
        self.display = display
        self.A = StatSpect.F1_values
        self.B = StatSpect.F2_values
        self.C = StatSpect.Data.loc[display].values
        self.manip_mode = StatSpect.manip_mode
        if dbk:
            self.dbk = dbk
        else:
            self.dbk = default_plot_settings(self.manip_mode)
        self.colormap = cmap
        self.slider_value = 3.0
        self.slider_start=0.05 
        self.slider_end=30.0
        self.slider_step=0.05
        if title:
            self.name = title
        else:
            self.name = StatSpect.name
        if levels is None:
            self.levels = np.array([0.5,1,2,5,10,20,50,100])
        else:
            self.levels = levels  
        xs,ys,col = BU.affiche_contour(self.A, self.B, self.C, cmap=self.colormap,levelbase=self.levels,
                                             scale=self.slider_value)
        if debug>1: print(xs,ys,col)
        self.source = ColumnDataSource(data=dict(xs=xs, ys=ys, color=col))
        self.plot = figure(**self.dbk, title=self.name )
        self.plot.multi_line(xs='xs', ys='ys', color='color', source=self.source)
        self.scale_slider = slider(title=self.name, value=self.slider_value, start=self.slider_start, end=self.slider_end, 
                      step=self.slider_step)
        self.scale_slider.on_change('value', self.update_data)
        self.widget = column(self.scale_slider,self.plot)
        if debug: print("plot for", StatSpect.name, display)
            
    def update_data(self, attr, old, new):
        # Get the current slider value & update source
        xs,ys,col = BU.affiche_contour( self.A, self.B, self.C, scale=self.scale_slider.value, cmap=self.colormap,levelbase=self.levels)
        self.source.data = dict(xs=xs, ys=ys, color=col)#update source
    
    def add_multiline(self, StatSpect2, display=None, title=None, cmap=None, levels=None):
        """
        add a multiline to the original plot and the associated slider.
        """
        if display is None:
            display = self.display
        new_data = BokehApp_Slider_Plot_mpl(StatSpect2, display, dbk=self.dbk, cmap=cmap, title=title, levels=levels)
        self.plot.multi_line(xs='xs', ys='ys', color='color', source=new_data.source)
        self.widget = column(new_data.scale_slider, self.widget)

class AnalysisPlots_mpl(object):
    """
    Class used to create Analysis plots, based on the BokehApp_Slider_Plot class. Needed data are different and the slider is not for scale but features.
    """
    def __init__(self, statSer, display, nfeatures, A, B, analysismode="RFE", dbk=None, cmap=None, title=None, levels=None):
        self.statSer = statSer
        self.display = display
        self.X = statSer.X(display)
        self.Y = self.statSer.activities
        self.D1 = statSer.data1.F1_values
        self.D2 = statSer.data1.F2_values
        self.manip_mode = statSer.data1.manip_mode
        if dbk:
            self.dbk = dbk
        else:
            self.dbk = default_plot_settings(self.manip_mode)
        self.analysismode = analysismode
        self.colormap = cmap
        if title is None:
            self.name = statSer.folder + ' / ' + analysismode
        else:
            self.name = title 
        self.slider_start = 0
        self.slider_end = 900
        self.slider_step = 5
        if levels is None:
            self.levels = np.array([0.5,1,2,5,10,20,50,100])
        else:
            self.levels = levels  
        #Building Sliders
        self.A_slider = slider(title="A", value=A, start=0.1, end=1, step=0.01)

        self.B_slider = slider(title="B", value=B, start=0.1, end=1, step=0.01)

        self.nfeatures_slider = slider(title="N Features Kept", value=nfeatures, start=self.slider_start, end=self.slider_end, step=self.slider_step)

        #Building sources: (self.source, self.source_correc_line, self.source_correc_points, self.source_chic)
        self.BuildDataAndSources()
        #Creating analysis plots according to chosen mode from sources
        self.plot = figure(**self.dbk, title=self.name )
        self.plot.multi_line(xs='xs', ys='ys', color='color', source=self.source)
        #Settings for control (correction) plot from sources
        TOOLS = "pan, box_zoom, undo, redo, reset, save"
        dbk_ctrl = {'tools': TOOLS}
        dbk_ctrl['x_axis_label'] = u"Concentration"
        dbk_ctrl['y_axis_label'] = u"Activity"
        dbk_ctrl['x_range'] = Range1d(-0.01, 1.01)
        dbk_ctrl['y_range'] = Range1d(-0.01, 1.01)
        self.plot_control = figure(**dbk_ctrl, title=self.name+" Y Correction")
        self.plot_control.line(x='line_x',y='line_y',line_width=2,source=self.source_correc_line)
        self.plot_control.circle(x='points_x',y='points_y',color='red', size=10, source=self.source_correc_points)
        self.plot_control.multi_line(xs='xs_ctrl',ys='ys_ctrl', color='coral',line_dash="dotdash", line_width=2, source=self.source_chic)
        self.A_slider.on_change('value', self.update_data)
        self.B_slider.on_change('value', self.update_data)
        self.nfeatures_slider.on_change('value', self.update_data)
        #Building complete widget
        self.widget = column(self.nfeatures_slider, self.A_slider, self.B_slider, self.plot, self.plot_control)

    def BuildDataForSourceChic(self):
        """
        Tool to build data for "source_chic" used in AnalysisPlots
        """
        xs_ctrl = []
        ys_ctrl = []
        for y in self.Y:
            tyy = ty(y, self.A_slider.value, self.B_slider.value)
            xs_ctrl.append([0, tyy, tyy])
            ys_ctrl.append([y, y, 0])
        return xs_ctrl, ys_ctrl

    def BuildDataForSourcesCorrection(self):
        """
        Tool to build the data for sources used in AnalysisPlots to display the correction of activities.
        """
        x = np.linspace(0.,1.,100)
        line_x = ty(x, self.A_slider.value, self.B_slider.value)
        points_x = ty(self.Y,self.A_slider.value, self.B_slider.value) #Corrected Y
        points_y = self.Y
        return line_x, x, points_x, points_y

    def ComputeData(self):
        ReY = ty(self.Y, self.A_slider.value, self.B_slider.value)
        if self.analysismode == "RFE":
            xs,ys,col = BU.affiche_contour(*(RecurFeatElim(self.X, ReY, self.D1, self.D2, nfeatures=self.nfeatures_slider.value)),
                                            cmap=self.colormap, levelbase=self.levels)
        elif self.analysismode == "LinReg":
            xs,ys,col = BU.affiche_contour(*(LinRegression(self.X, ReY, self.D1, self.D2, nfeatures=self.nfeatures_slider.value)), 
                                            cmap=self.colormap, levelbase=self.levels)
        elif self.analysismode == "LogisticRegr":
            xs,ys,col = BU.affiche_contour(*(LogisticRegr(self.X, ReY, self.D1, self.D2, nfeatures=self.nfeatures_slider.value)), 
                                            cmap=self.colormap, levelbase=self.levels)
        elif self.analysismode == "RidgeReg":
            xs,ys,col = BU.affiche_contour(*(ridge(self.X, ReY, self.D1, self.D2, nfeatures=self.nfeatures_slider.value)), 
                                            cmap=self.colormap, levelbase=self.levels)
        elif self.analysismode == "ElasticNet":
            xs,ys,col = BU.affiche_contour(*(elasticnet(self.X, ReY, self.D1, self.D2, nfeatures=self.nfeatures_slider.value)), 
                                            cmap=self.colormap, levelbase=self.levels)
        else:
            raise Exception("WRONG MODE")
        return xs,ys,col

    def BuildDataAndSources(self): # mode, X, Y, x, A, B, D1, D2, nfeatures, cmap, levelbase):
        """
        Build all the sources necessary in AnalysisPlots
        """
        xs,ys,col = self.ComputeData()
        line_x,line_y,points_x,points_y = self.BuildDataForSourcesCorrection()
        xs_ctrl,ys_ctrl = self.BuildDataForSourceChic()
        self.source = ColumnDataSource(data=dict(xs=xs, ys=ys, color=col))
        self.source_correc_line = ColumnDataSource(data=dict(line_x=line_x, line_y=line_y))
        self.source_correc_points = ColumnDataSource(data=dict(points_x=points_x, points_y=points_y))
        self.source_chic = ColumnDataSource(data=dict(xs_ctrl=xs_ctrl, ys_ctrl=ys_ctrl))

    def update_data(self, attr, old, new):
        """
        Get the current slider value & update source
        """
        xs,ys,col = self.ComputeData()
        line_x,line_y,points_x,points_y = self.BuildDataForSourcesCorrection()
        xs_ctrl,ys_ctrl = self.BuildDataForSourceChic()
        self.source.data = dict(xs=xs, ys=ys, color=col)
        self.source_correc_line.data = dict(line_x=line_x, line_y=line_y)
        self.source_correc_points.data = dict(points_x=points_x, points_y=points_y)
        self.source_chic.data = dict(xs_ctrl=xs_ctrl, ys_ctrl=ys_ctrl)

    def add_multiline(self, StatSpect2, display=None, title=None, cmap=None, levels=None):
        """
        add a multiline to the original plot and the associated slider.
        """
        if display is None:
            display = self.display
        new_data = BokehApp_Slider_Plot_mpl(StatSpect2, display, dbk=self.dbk, cmap=cmap, title=title, levels=levels)
        self.plot.multi_line(xs='xs', ys='ys', color='color', source=new_data.source)
        self.widget = column(new_data.scale_slider, self.widget)

def mpl_create_app(doc, folder, data_name, data_name2, activities, display=["std"], manip_mode='TOCSY', dataref=None, normalize=False, netmode='mieux', correction=True, B=0.5, A=0.1,sym=True,net=True,nfeatures=100, debug=False):
    """
    This function creates the complete bokeh application for Plasmodesma results analysis.
    - doc is the current document in which the app is made.
    - folder is the folder in which the results of plasmodesma are stored.
    - activities: the activites corresponding to the different fractions present in the folder.
    - name is the name of the first data to display & analyse.
    - name2 is the name of the second data to display & analyse.
    - manip_mode: 3 modes available: HSQC,TOCSY,COSY: determines which experiments are loaded
    - netmode is the cleaning/denoising mode desired for plasmodesma (BucketUtilities).
    """
    BU.NETMODE = netmode

    FullData = BU.StatSeries(folder, data_name, data_name2, activities, manip_mode=manip_mode, dataref=dataref, sym=sym, net=net, normalize=normalize)   # load all datasets
    if debug:
        print( FullData.activities )
        print( FullData.data1.epath )
        print( FullData.Series[0].Data)
        if debug > 1:
            return FullData

    Graph1 =  BokehApp_Slider_Plot_mpl(FullData.data1, display[0], debug=debug)
    Graph2 =  BokehApp_Slider_Plot_mpl(FullData.data2, display[0], dbk=Graph1.dbk, debug=debug)
    DataRatio = copy(FullData.data1)
    DataRatio.name = "Ratio"
    DataRatio.Data = FullData.data1.Data/(FullData.data2.Data+1e-5)
    DataDiff = copy(FullData.data1)
    DataDiff.name = "Difference"
    DataDiff.Data = FullData.data1.Data - FullData.data2.Data
    GraphRatio = BokehApp_Slider_Plot_mpl(DataRatio, display[0], dbk=Graph1.dbk, debug=debug)
    GraphSubstract = BokehApp_Slider_Plot_mpl(DataDiff, display[0], dbk=Graph1.dbk, debug=debug)

    if not correction: #A way to tell t(y) to do nothing
        A = 0
        B = 0

    GraphRFE =  AnalysisPlots_mpl(FullData, display[0], nfeatures, A, B, analysismode="RFE", dbk=Graph1.dbk)
    GraphLinReg = AnalysisPlots_mpl(FullData, display[0], nfeatures, A, B, analysismode="LinReg", dbk=Graph1.dbk)
    GraphRidge = AnalysisPlots_mpl(FullData, display[0], nfeatures, A, B, analysismode="RidgeReg", dbk=Graph1.dbk)
    Graphelasticnet = AnalysisPlots_mpl(FullData, display[0], nfeatures, A, B, analysismode="ElasticNet", dbk=Graph1.dbk)
    # GraphLogistReg =  AnalysisPlots(X=X,Y=Y,D1=Im1, D2=Im2, nfeatures=nfeatures,manip_mode=manip_mode, mode="LogisticRegr",dbk=Graph1.dbk, title="Logisitic Regression")


    if dataref is not None:
        i=0
        colors = [cm.autumn,cm.summer,cm.spring,cm.winter,cm.GnBu]
        for i in range(0,len(dataref)):
            GraphRatio.add_multiline(FullData.reference[i], display='bucket', title=dataref[i], cmap=colors[i], levels=[1])
            GraphSubstract.add_multiline(FullData.reference[i], display='bucket', title=dataref[i], cmap=colors[i], levels=[1])
            GraphRFE.add_multiline(FullData.reference[i], display='bucket', title=dataref[i], cmap=colors[i], levels=[1])
            GraphLinReg.add_multiline(FullData.reference[i], display='bucket', title=dataref[i], cmap=colors[i], levels=[1])
            GraphRidge.add_multiline(FullData.reference[i], display='bucket', title=dataref[i], cmap=colors[i], levels=[1])
            Graphelasticnet.add_multiline(FullData.reference[i], display='bucket', title=dataref[i], cmap=colors[i], levels=[1])
        #    GraphLogistReg.add_multiline(FullData.reference, display='bucket', title="Reference", cmap=cm.autumn, levels=[1])

    # Set up layouts and add to document
    tab1 = Panel(child=column(row(Graph1.widget, Graph2.widget),row(GraphRatio.widget, GraphSubstract.widget)), 
                 title="Visualization")
    tab2 = Panel(child=column(row(GraphRFE.widget, GraphLinReg.widget), row(GraphRidge.widget,Graphelasticnet.widget)), title="Global Analysis")
    
    doc.add_root(Tabs(tabs=[tab1, tab2]))
    doc.title = folder
    doc.theme = Theme(json=yaml.load("""
        attrs:
            Figure:
                background_fill_color: "#DDDDDD"
                outline_line_color: white
                toolbar_location: right
            Grid:
                grid_line_dash: [6, 4]
                grid_line_color: white
    """))
    if debug: print("et voila")
