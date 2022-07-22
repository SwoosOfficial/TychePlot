
# coding: utf-8

# In[63]:

import numpy as np
import scipy.optimize as sco
import copy
import warnings
from Data import Data


# In[94]:

class Fitter:
    
    default_fitColors=[
            "#000000",
            "#1f77b4",
            "#d62728",
            "#2ca02c",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#ff7f0e",
            "#bcbd22",
            "#17becf",
            "#f8e520",
        ]
    
    @classmethod
    def return_completed_fitter(cls, 
                                data,
                                fTuple,
                                noFit=False,
                                **kwargs
                                ):
                    
        fitter=Fitter(
                    data,
                    fTuple[2],
                    errorData=None,
                    dataForFitXLim=fTuple[0],
                    dataForFitYLim=fTuple[6].pop("dataForFitYLim", None),
                    curveDataXLim=fTuple[1],
                    params=fTuple[3],
                    textPos=fTuple[4],
                    desc=fTuple[5],
                    addKwArgs=fTuple[6],
                    **kwargs
        )
        if isinstance(fitter.dataForFitXLim[0], list):
            feature = [11, 2]
        else:
            feature = [1, 2]
        if isinstance(fitter.curveDataXLim[0], list):
            feature[1] = 21
        else:
            feature[1] = 2
        fitter.limitData(xLim=fitter.dataForFitXLim, yLim=fitter.dataForFitYLim, feature=feature[0])
        if not noFit:
            try:
                fitter.fit(xCol=fitter.xCol, yCol=fitter.yCol, p0=fitter.params)
            except RuntimeError as err:
                raise cls.FitException(
                    fitter.function,
                    "",
                    err,
                    fitter.params,
                )
        fitter.doFitCurveData(xCol=fitter.data.xCol)
        fitter.limitData(xLim=fitter.curveDataXLim, feature=feature[1])
        return fitter
    
    def __init__(self, data, function,
                 errorData=None,
                 params = None,
                 dataXLim=None,
                 dataForFitXLim=None,
                 curveDataXLim=None,
                 dataYLim=None,
                 dataForFitYLim=None,
                 curveDataYLim=None,
                 textPos=None,
                 desc=None,
                 xCol=0,
                 yCol=0,
                 fitColors=default_fitColors,
                 fitAlpha=0.75,
                 fitLs=":",
                 addKwArgs={}):
        self.data = data
        self.function = function
        self.dataForFit = copy.deepcopy(data)
        self.params = params
        self.params_cov = None
        self.CurveData = None
        self.errorData = errorData
        self.dataXLim=dataXLim
        self.dataForFitXLim=dataForFitXLim
        self.curveDataXLim=curveDataXLim
        self.dataYLim=dataYLim
        self.dataForFitYLim=dataForFitYLim
        self.curveDataYLim=curveDataYLim
        self.textPos=textPos
        self.desc=desc
        self.xCol=xCol
        self.yCol=yCol
        self.fitColors=fitColors
        self.fitAlpha=fitAlpha
        self.fitLs=fitLs
        self.addKwArgs=addKwArgs
        
    def limitInputData(self, xLim=None, yLim=None):
        """deprecated"""
        self.dataForFit.limitData(xLim=xLim, yLim=yLim)
        
    def limitData(self, xLim=None, yLim=None, feature=0):
        if feature == -1:
            self.data.limitData(xLim=self.dataXLim, yLim=self.dataYLim)
            self.dataForFit.limitData(xLim=self.dataForFitXLim, yLim=self.dataForFitYLim)
            self.CurveData.limitData(xLim=self.curveDataXLim, yLim=self.curveDataYLim)
        elif feature == 0:
            self.data.limitData(xLim=xLim, yLim=yLim)
        elif feature == 1:
            if self.errorData is not None:
                if yLim is not None:
                    warnings.warn("YLim with errors not yet implemented")
                    self.dataForFit.limitData(yLim=yLim)
                if xLim[0]<=xLim[1]:
                    n=self.dataForFit.getFirstIndexWhereGreaterOrEq(self.dataForFit.xCol,xLim[0])
                    m=self.dataForFit.getLastIndexWhereSmallerOrEq(self.dataForFit.xCol,xLim[1])
                    if m==-1:
                        self.dataForFit.setData(self.dataForFit.getData()[n:])
                        self.errorData.setData(self.errorData.getData()[n:])
                    else:
                        self.dataForFit.setData(self.dataForFit.getData()[n:m+1])
                        self.errorData.setData(self.errorData.getData()[n:m+1])
                else:
                    n=self.dataForFit.getFirstIndexWhereSmallerOrEq(self.dataForFit.xCol,xLim[0])
                    m=self.dataForFit.getLastIndexWhereGreaterOrEq(self.dataForFit.xCol,xLim[1])
                    if m==-1:
                        self.dataForFit.setData(self.dataForFit.getData()[n:])
                        self.errorData.setData(self.errorData.getData()[n:])
                    else:
                        self.dataForFit.setData(self.dataForFit.getData()[n:m+1])
                        self.errorData.setData(self.errorData.getData()[n:m+1])
            else:
                self.dataForFit.limitData(xLim=xLim, yLim=yLim)
        elif feature == 2:
            try:
                self.CurveData.limitData(xLim=xLim, yLim=yLim)
            except IndexError:
                self.CurveData.limitData(xLim=(xLim[1],xLim[0]), yLim=yLim)
        elif feature == 11:
            if xLim[0][0]<=xLim[0][1]:
                self.dataForFit.intersectData(xLim)
            else:
                self.dataForFit.intersectData(xLim, invert_intervals=True)
        elif feature == 21:
            if xLim[0][0]<=xLim[0][1]:
                self.CurveData.intersectData(xLim)
            else:
                self.CurveData.intersectData(xLim, invert_intervals=True)
            
        else:
            raise ValueError("No Such Feature")
    
    def fit(self,yCol=None, xCol=None, **kwargs):
        if "p0" not in kwargs:
            kwargs.update({"p0":self.params})
        try:
            if self.errorData is not None and "sigma" not in kwargs:
                sigma=self.errorData.getSplitData2D(yCol=yCol)[1]
                if np.all((sigma,np.zeros(len(sigma)))):
                    kwargs.update({"sigma":sigma, "absolute_sigma":True}) 
            kwargs.update(self.addKwArgs)
            self.params, self.params_cov = sco.curve_fit(self.function, *self.dataForFit.getSplitData2D(xCol=xCol, yCol=yCol), **kwargs)
        except (NameError, TypeError):
            if self.errorData is not None and "sigma" not in kwargs:
                sigma=self.errorData.getSplitData2D(yCol=self.data.yCol)[1]
                if np.all((sigma,np.zeros(len(sigma)))):
                    kwargs.update({"sigma":sigma, "absolute_sigma":True}) 
            kwargs.update(self.addKwArgs)
            self.params, self.params_cov = sco.curve_fit(self.function, *self.dataForFit.getSplitData2D(xCol=self.data.xCol, yCol=self.data.yCol), **kwargs)
        
        
    def doFitCurveData(self, xCol=1, x_padding_fac = 0.05, num=10**4, xmax=None, xmin=None):
        xdata = self.data.getSplitData2D(xCol=xCol)[0]
        if xmin==None:
               x_min = np.amin(xdata)
        else:
               x_min = xmin
        if xmax==None:
               x_max = np.amax(xdata)
        else:
              x_max = xmax
        d_x = x_max-x_min
        x_padding = x_padding_fac*d_x
        fit_x_data = np.linspace(x_min-x_padding/2, x_max+x_padding/2, num=num)
        fit_y_data = self.function(fit_x_data, *self.params)
        self.CurveData = Data.initData2D(fit_x_data, fit_y_data)

class FitException(Exception):
    def __init__(self, index, props, err, params):
        self.message = (
            "Error @ {} with Properties: {} \n Message: {} with parameters {}".format(
                index, props, str(err), str(params)
            )
        )