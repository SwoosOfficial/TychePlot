# coding: utf-8

import matplotlib as mpl
mpl.use("pgf")
import matplotlib.pyplot
import numpy as np
import scipy.interpolate as inter
import copy
import sys
import os
import string
import warnings
import functools
import copy
import inspect
from matplotlib import rc
from Filereader import fileToNpArray
from Data import Data
from Fitter import Fitter
from Plot import Plot





class ProfiloPlot(Plot):

    @classmethod
    def linear(cls, x, a, b):
        return a*x+b

    @classmethod
    def AAToNm(cls, x):
        return x/10
    
    
    
    def __init__(self,
                 name,
                 fileList,
                 fileFormat={"separator":",", "skiplines":40, "backoffset":2, "lastlines":2},
                 title=None,
                 validYCol=[2],
                 showColAxType=["lin","lin","lin"],
                 showColAxLim=[None,None,None],
                 showColLabel= ["","Length","Height"],
                 showColLabelUnit=["",
                  "Length (µm)",
                  "Height (nm)"
                 ],
                 averageMedian=False,
                 errors=False,
                 xParamPos=0,
                 modFitTup=(0,0),
                 refFitTup=(0,1),
                 noMod=True,
                 **kwargs
                ):
        Plot.__init__(self, name, fileList, averageMedian=averageMedian, showColAxType=showColAxType, showColAxLim=showColAxLim, showColLabel=showColLabel, showColLabelUnit=showColLabelUnit, fileFormat=fileFormat, errors=errors, **kwargs)
        #dyn inits
        if title is None:
            self.title=name
        else:
            self.title=title
        self.xParamPos=xParamPos
        self.modFitTup=modFitTup
        self.refFitTup=refFitTup
        self.noMod=noMod
        #if self.noMod:
            #self.fitList=[self.fitList[self.modFitTup[0]][self.modFitTup[1]]]
           
   
    def offset(self, x):
        if self.xLimOrig is None:
            return x
        return x-self.xLimOrig[0]

    def processFileName(self, option=".pdf"):
        if self.filename is None:
            string=self.name.replace(" ","")+self.fill+"topology"
        else:
            string=self.filename
        if not self.scaleX is 1:
            string+=self.fill+"scaledWith{:03.0f}Pct".format(self.scaleX*100)
        return string+option
    
    def processData(self):
        if not self.dataProcessed:
            for device in self.dataList:
                for data in device:
                    data.limitData(xLim=self.xLimOrig)
                    data.processData(self.AAToNm)
                    data.processData(self.offset, x=True, y=False)
            self.dataProcessed=True
        return self.dataList
    
    def modFunc(self, ydata, xdata=None, params=None):
        return ydata-self.linear(xdata, *params)
        
    
    def modifyAllData(self):
        for data in self.expectData:
            data.modifyData(self.modFunc, params=self.fitterList[self.modFitTup[0]][self.modFitTup[1]].params)
        if self.fitList is None:
            return
        for fitter in self.fitterList:
            if fitter is not None:
                if type(fitter) is list:
                    for subFitter in fitter:
                        #subFitter.data.modifyData(self.modFunc, params=self.fitterList[self.modFitTup[0]][self.modFitTup[1]].params)
                        subFitter.dataForFit.modifyData(self.modFunc, params=self.fitterList[self.modFitTup[0]][self.modFitTup[1]].params)
                        subFitter.CurveData.modifyData(self.modFunc, params=self.fitterList[self.modFitTup[0]][self.modFitTup[1]].params)
                else:
                    #fitter.data.modifyData(self.modFunc, params=self.fitterList[self.modFitTup[0]][self.modFitTup[1]].params)
                    fitter.dataForFit.modifyData(self.modFunc, params=self.fitterList[self.modFitTup[0]][self.modFitTup[1]].params)
                    fitter.CurveData.modifyData(self.modFunc, params=self.fitterList[self.modFitTup[0]][self.modFitTup[1]].params)
        self.refit()
        
    def refit(self):
        if self.fitList is None:
            return
        for fitter in self.fitterList:
            if fitter is not None:
                if type(fitter) is list:
                    for subFitter in fitter:
                        if subFitter.dataForFitYLim is None:
                            yLim=None
                        else:
                            yLim=subFitter.dataForFitYLim-self.fitterList[self.modFitTup[0]][self.modFitTup[1]].params[1]
                        subFitter.limitData(xLim=subFitter.dataForFitXLim, yLim=yLim , feature=1)
                        try:
                            subFitter.fit(xCol=self.xCol, yCol=self.showCol, p0=subFitter.params)
                        except RuntimeError as err:
                            raise FitException(self.fitterList.index(fitter),fitList[(self.fitterList.index(fitter))],err)
                        subFitter.doFitCurveData()
                        subFitter.limitData(xLim=subFitter.curveDataXLim, feature=2)
                else:
                    if fitter.dataForFitYLim is None:
                        yLim=None
                    else:
                        yLim=fitter.dataForFitYLim-self.fitterList[self.modFitTup[0]][self.modFitTup[1]].params[1]
                    fitter.limitData(xLim=fitter.dataForFitXLim,yLim=yLim, feature=1)
                    try:
                        fitter.fit(xCol=self.xCol, yCol=self.showCol, p0=fitter.params)
                    except RuntimeError as err:
                        raise FitException(self.fitterList.index(fitter),self.fitList[(self.fitterList.index(fitter))],err,fitter.params)
                    fitter.doFitCurveData(xCol=self.xCol)
                    fitter.limitData(xLim=fitter.curveDataXLim, feature=2)
    
    def processPlot(self):
        expectData=self.processAverage()[0]
        colors=self.colors
        labels=self.labels
        xCol=1
        showCol=2
        ax=self.ax
        if self.fitList is not None:
            horiFitter=self.fitterList[self.refFitTup[0]][self.refFitTup[1]]
            botFitter=self.fitterList[self.modFitTup[0]][self.modFitTup[1]]
            annoPos=horiFitter.textPos
            if not self.noMod:
                self.modifyAllData()
            self.delta = np.absolute(int(np.round(horiFitter.function(annoPos[0], *horiFitter.params)-botFitter.function(annoPos[0], *botFitter.params), decimals=0)))
        for n in range(0,len(expectData)):
            if self.show[n][0]:
                AX=ax.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=showCol), c=colors[n], ls=self.ls, label=labels[n])
                if self.fitList is not None and self.fitterList[n] is not None:
                    if type(self.fitterList[n]) is list:
                        for fitter in self.fitterList[n]:
                            FIT=ax.errorbar(*fitter.CurveData.getSplitData2D(xCol=xCol, yCol=showCol), c=self.fitColors[n], ls=self.fitLs, label=self.fitLabels[n], alpha=self.fitAlpha)    
                    else:
                        FIT=ax.errorbar(*self.fitterList[n].CurveData.getSplitData2D(xCol=xCol, yCol=showCol), c=self.fitColors[n], ls=self.fitLs, label=self.fitLabels[n], alpha=self.fitAlpha)
        if self.fitList is not None:
            if self.noMod:
                ax.annotate(s="", xy=(annoPos[0],botFitter.function(annoPos[0], *botFitter.params)), xytext=(annoPos[0],horiFitter.function(annoPos[0], *horiFitter.params)), arrowprops=dict(arrowstyle="<->"))
                ax.annotate(s=str(self.delta)+"\\,nm", xy=(annoPos[1], botFitter.function(annoPos[0], *botFitter.params)/3))
            else:
                ax.annotate(s="", xy=(annoPos[0],0), xytext=(annoPos[0],horiFitter.function(annoPos[0], *horiFitter.params)), arrowprops=dict(arrowstyle="<->",linewidth=mpl.rcParams["lines.linewidth"]))
                ax.annotate(s=str(self.delta)+"\\,nm", xy=(annoPos[1], horiFitter.function(annoPos[0], *horiFitter.params)/3))
                        
        
   



    
    
   