"""
This program contains all useful functions and classes used to create the bokeh app for plasmodesma results display.

"""
import os
import yaml
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

import BucketUtilities as BU

def ty(x, A, B):
    "A defines the slope of the sigmoid"
    if A+B == 0 :
        return x
    val = B + A*np.arctanh(2*x-1)
    val = np.minimum(val,1.0)
    val = np.maximum(val,0.0)
    return val

def prepare_analysis(results_folder, ref_name, extension="2D/dipsi2phpr_20_bucketlist.csv", sym=True, net=True):
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
            Int = BU.loadInt2D(tit, net=net, sym=sym)
            tInt = Int[2].ravel() # [tmask]
            X.append(tInt)
    X = np.array(X)
    t = np.linspace(0, 1, 100)
    return X

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
    return (Im1[0], Im1[1], m)

def RecurFeatElim(X, Y, Im1, Im2, nfeatures=100):
    """
    Performs RFE analysis from scikit learn between name and name2 datasets.
    - X,Y are the results of prepare_analysis.
    - name and name2 are the name of data to be displayed on which analyse is performed.
    return a mplt axis object(contour_plot) which is then passed to get_contour_data to be displayed with bokeh.
    """
    estimator = linear_model.LinearRegression()
    selector = RFE(estimator, step=0.5, n_features_to_select=nfeatures)
    selector = selector.fit(X, Y)
    N = len(Im2[2].ravel())
    m = np.zeros(N)
    m[selector.support_] = 1.0
    m = m.reshape(Im2[2].shape)
    return (Im1[0], Im1[1], m)

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
    if manip_mode in ("TOCSY","COSY"):
        dbk['y_range'] = Range1d(10, 0)
    elif manip_mode == "HSQC":
        dbk['y_range'] = Range1d(150, 0)
    else:
        print("No compatible mode chosen!")
    return dbk

def slider(title, value, start, end, step):
    """
    Creates a Bokeh Slider
    """
    return Slider(title=title, value=value, start=start, end=end, 
                      step=step)

class BokehApp_Slider_Plot(object):
    """
    Class used to create plot and associated slider from loadStd2D() results.
    """
    def __init__(self, A, B, C, manip_mode='TOCSY', dbk=None, cmap=None, title="my name", levels=[0.5,1,2,5,10,20,50,100]):
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
        self.slider_start = 0.05 
        self.slider_end = 30.0
        self.slider_step = 0.05
        self.name = title
        self.levels = levels  
        xs,ys,col = BU.affiche_contour(self.A, self.B, self.C, cmap=self.colormap, levelbase=self.levels,
                                             scale=self.slider_value)
        self.source = ColumnDataSource(data=dict(xs=xs, ys=ys, color=col))
        self.plot = figure(**self.dbk, title=self.name )
        self.plot.multi_line(xs='xs', ys='ys', color='color', source=self.source)
        self.scale_slider = slider(title=self.name, value=self.slider_value, start=self.slider_start, 
                                        end=self.slider_end, step=self.slider_step)
        self.scale_slider.on_change('value', self.update_data)
        self.widget = column(self.scale_slider, self.plot)
            
    def update_data(self, attr, old, new):
        # Get the current slider value & update source
        xs,ys,col = BU.affiche_contour(self.A, self.B, self.C, scale=self.scale_slider.value, 
                                        cmap=self.colormap, levelbase=self.levels)
        self.source.data = dict(xs=xs, ys=ys, color=col)#update source
    
    def add_multiline(self, E, F, G, manip_mode, title, cmap, levels):
        """
        add a multiline to the original plot and the associated slider.
        """
        new_data = BokehApp_Slider_Plot(E, F, G, dbk=self.dbk, manip_mode=manip_mode, 
                                        cmap=cmap, title=title, levels=levels)
        self.plot.multi_line(xs='xs', ys='ys', color='color', source=new_data.source)
        self.widget = column(new_data.scale_slider, self.scale_slider, self.plot)


class AnalysisPlots(object):
    """
    Class used to create Analysis plots, based on the BokehApp_Slider_Plot class. Needed data are different and the slider is not for scale but features.
    """
    def __init__(self, X, Y, D1, D2, nfeatures, A, B, manip_mode='TOCSY', dbk=None, cmap=None, title="my name", levels=[0.5,1,2,5,10,20,50,100], mode="RFE"):
        self.X = X
        self.Y = Y
        self.D1 = D1
        self.D2 = D2
        self.manip_mode = manip_mode
        if dbk:
            self.dbk = dbk
        else:
            self.dbk = default_plot_settings(self.manip_mode)
        self.mode = mode
        self.colormap = cmap
        self.name = title 
        self.slider_start = 0
        self.slider_end = 500
        self.slider_step = 5
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
        if self.mode == "RFE":
            xs,ys,col = BU.affiche_contour(*(RecurFeatElim(self.X, ReY, self.D1, self.D2, nfeatures=self.nfeatures_slider.value)),
                                            cmap=self.colormap, levelbase=self.levels)
        elif self.mode == "LinReg":
            xs,ys,col = BU.affiche_contour(*(LinRegression(self.X, ReY, self.D1, self.D2, nfeatures=self.nfeatures_slider.value)), 
                                            cmap=self.colormap, levelbase=self.levels)
        elif self.mode == "LogisticRegr":
            xs,ys,col = BU.affiche_contour(*(LogisticRegr(self.X, ReY, self.D1, self.D2, nfeatures=self.nfeatures_slider.value)), 
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

    def add_multiline(self, E, F, G, manip_mode, title, cmap, levels):
        """
        add a multiline to the original plot and the associated scale slider.
        """
        new_data = BokehApp_Slider_Plot(E, F, G, dbk=self.dbk, manip_mode=manip_mode, cmap=cmap, title=title, levels=levels)
        self.plot.multi_line(xs='xs', ys='ys', color='color', source=new_data.source)
        self.widget = column(new_data.scale_slider, self.nfeatures_slider, self.A_slider, self.B_slider, self.plot, self.plot_control)

def new_create_app(doc, folder, dataref, data_name, data_name2, netmode, activities, manip_mode, extension="2D/dipsi2phpr_20_bucketlist.csv", correction=True, B=0.5, A=0.1, sym=True, net=True, loadwith="PP2D", nfeatures=100):
    """
    This function creates the complete bokeh application for Plasmodesma results analysis.
    - doc is the current document in which the app is made.
    - folder is the folder in which the results of plasmodesma are stored.
    - activities: the activites corresponding to the different fractions present in the folder.
    - name is the name of the first data to display & analyse.
    - name2 is the name of the second data to display & analyse.
    - netmode is the cleaning/denoising mode desired for plasmodesma (BucketUtilities).
    """
    BU.NETMODE = netmode
    name = os.path.join(folder, data_name)
    name2 = os.path.join(folder, data_name2)
    refname = os.path.join(folder, dataref)
    def load(loadwith, name, net=net, sym=sym):
        """
        Used to decide which type of loading(from bucketutilities) is used.
        """
        if loadwith == "Std2D":
            return BU.loadStd2D(name, net=net, sym=sym)
        if loadwith == "PP2D":
            return BU.loadPP2D(name, net=net, sym=sym)
    Im1 = load(loadwith=loadwith, name=name, net=net, sym=sym)
    Im2 = load(loadwith=loadwith, name=name2, net=net, sym=sym)
    Imref = load(loadwith="Std2D", name=refname, net=net, sym=sym)
    X = prepare_analysis(folder, dataref, extension=extension, sym=sym, net=net)
    
    Graph1 =  BokehApp_Slider_Plot(*Im1, manip_mode=manip_mode, title=data_name)
    Graph2 =  BokehApp_Slider_Plot(*Im2, dbk=Graph1.dbk, manip_mode=manip_mode, title=data_name2)
    GraphRatio =  BokehApp_Slider_Plot(Im1[0], Im1[1], Im1[2]/(Im2[2]+1e-5), dbk=Graph1.dbk, manip_mode=manip_mode, 
                                         title="Ratio")
    GraphRatio.add_multiline(*Imref, manip_mode=manip_mode, title="Reference", cmap=cm.autumn, levels=[1])
    GraphSubstract =  BokehApp_Slider_Plot(Im1[0], Im1[1], Im1[2]-Im2[2], dbk=Graph1.dbk, manip_mode=manip_mode, 
                                             title="Subtraction")
    GraphSubstract.add_multiline(*Imref, manip_mode=manip_mode, title="Reference", cmap=cm.autumn, levels=[1])
    if not correction: #A way to tell t(y) to do nothing
        A = 0
        B = 0

    GraphRFE =  AnalysisPlots(X=X, Y=activities, A=A, B=B, D1=Im1, D2=Im2, nfeatures=nfeatures, manip_mode=manip_mode, mode="RFE", dbk=Graph1.dbk, title="RFE")
    GraphRFE.add_multiline(*Imref, manip_mode=manip_mode, title="Reference", cmap=cm.autumn, levels=[1])
    
    GraphLinReg =  AnalysisPlots(X=X, Y=activities, A=A, B=B, D1=Im1, D2=Im2, nfeatures=nfeatures, manip_mode=manip_mode, mode="LinReg",dbk=Graph1.dbk, title="Linear Regression")
    GraphLinReg.add_multiline(*Imref, manip_mode=manip_mode, title="Reference", cmap=cm.autumn, levels=[1])

    # GraphLogistReg =  AnalysisPlots(X=X,Y=Y,D1=Im1, D2=Im2, nfeatures=nfeatures,manip_mode=manip_mode, mode="LogisticRegr",dbk=Graph1.dbk, title="Logisitic Regression")
    # GraphLogistReg.add_multiline(*Imref,manip_mode=manip_mode,title="Reference",cmap=cm.autumn, levels=[1])
    
    # Set up layouts and add to document
    tab1 = Panel(child=column(row(Graph1.widget, Graph2.widget),row(GraphRatio.widget, GraphSubstract.widget)), 
                 title="Visualization")
    tab2 = Panel(child=column(row(GraphRFE.widget, GraphLinReg.widget)), title="Global Analysis")
    
    doc.add_root(Tabs(tabs=[tab1, tab2]))
    
    doc.title = "SMARTE"

    doc.theme = Theme(json=yaml.load("""
        attrs:
            Figure:
                background_fill_color: "#DDDDDD"
                outline_line_color: white
                toolbar_location: right
                height: 450
                width: 450
            Grid:
                grid_line_dash: [6, 4]
                grid_line_color: white
    """))
