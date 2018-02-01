
# coding: utf-8

# In[1]:

texPreamble=[r"\usepackage{amsmath}",
             r"\usepackage{fontspec}",
             r"\usepackage[no-sscript]{xltxtra}",
             r"\setmainfont{Minion-Pro_Regular.ttf}[BoldFont = Minion-Pro-Bold.ttf, ItalicFont = Minion-Pro-Italic.ttf, BoldItalicFont = Minion-Pro-Bold-Italic.ttf]"]
pgfSys="xelatex"

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


# In[2]:


class Plot():
    
    @classmethod
    def equalizeRanges(cls, data, norm=(390,780,401)):
        arr, arr2=data.getSplitData2D()
        f=inter.CubicSpline(arr,arr2,extrapolate=True)
        x=np.linspace(*norm)
        data.setData(Data.mergeData((x,f(x))))
    
    @classmethod
    def div0(cls, a, b ):
        """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide( a, b )
            c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
        return c
    
    @classmethod
    def normalize2(cls,a):
        return a/np.amax(a)
    
    @classmethod
    def normalize(cls,a):
        return a/np.sum(a)
    
    @classmethod
    def absolute(cls,a):
        return np.absolute(a)
    
    @classmethod
    def initByExistingPlot(cls, obj, **kwargs):
        return cls(obj.name,
                   obj.fileList,
                   dataList=obj.dataList,
                   errList=[obj.expectData,obj.deviaData,obj.logErr],
                   dataProcessed=True,
                   averageProcessed=True,
                   dataImported=True,
                   **kwargs
                   )
    
    
    def __initFileList(self, fileList, errors, labels, show):
        if self.overrideFileList:
            self.fileList=[]
            self.errors=[]
            self.labels=[]
            self.show=[]
        else:
            try:
                temp=fileList[0][0]
                self.fileList=fileList
            except IndexError:
                raise ListShapeException("The filelist has to be formatted like: [[sampleApx1,sampleApx2],[sampleBpx1,sampleBpx2]]")
            if errors is None and labels is None:
                self.show=[[True,True] for device in self.fileList]
                self.errors=self.show
                self.labels=["Sample {:d}".format(m+1) for m in range(0,len(self.fileList))]
            elif labels is not None:
                try:
                    temp=[labels[m] for m in range(0,len(self.fileList))]
                    self.labels=labels
                except IndexError:
                    raise ListShapeException("The labels' list has to be formatted like: [\"sampleALabel\",\"sampleBLabel\"]")
                self.errors=[[deviceLabel is not "", deviceLabel is not ""] for deviceLabel in self.labels]
                self.show=self.errors
            else:
                try:
                    if self.showCol is 0 or self.showCol2 is 0:
                        temp=[errors[m][0] for m in range(0,len(self.fileList))]
                    else:
                        temp=[errors[m][1] for m in range(0,len(self.fileList))]
                    self.errors=errors
                except IndexError:
                    raise ListShapeException("The errors' list has to be formatted like: [[sampleAerrorAxis1,sampleAerrorAxis2],[sampleBerrorAxis1,sampleBerrorAxis2]]")
                self.labels=["Sample {:d}".format(m+1) for m in range(0,len(self.fileList))]
                self.show=self.errors
            if show is not None:
                try:
                    if self.showCol is 0 or self.showCol2 is 0:
                        temp=[show[m][0] for m in range(0,len(self.fileList))]
                    else:
                        temp=[show[m][1] for m in range(0,len(self.fileList))]
                    self.show=show
                except IndexError:
                    raise ListShapeException("The show list has to be formatted like: [[sampleAshowAxis1,sampleAshowAxis2],[sampleBshowAxis1,sampleBshowAxis2]]")
            if errors is not None:
                if errors is False:
                    self.errors=[[False,False] for device in self.fileList]
                elif errors is True:
                    self.errors=[[True,True] for device in self.fileList]
                elif errors[0] is True and errors[1] is False:
                    self.errors=[[True,False] for device in self.fileList]
                elif errors[0] is False and errors[1] is True:
                    self.errors=[[True,False] for device in self.fileList]

    def __init__(self,
                 name,
                 fileList,
                 fileFormat={"separator":"\t", "skiplines":1},
                 showColTup=(2,3),
                 xLim=None,
                 xLimOrig=1,
                 yLim=None,
                 scaleX=1,
                 customFontsize=None,
                 averageMedian=False,
                 xCol=1,
                 xColOrig=1,
                 title=None,
                 HWratio=3/4, # height to width ratio
                 fig_width_pt=448.13095, # get it by \the\textwidth
                 titleBool=True,
                 legendEdgeSize=1,
                 ax2Labels=True,
                 spectralDataformat={"separator":";", "skiplines":75}, #jeti csv format
                 showColAxType=["","lin","lin","lin"],
                 showColAxLim=["",None,None,None],
                 colors=['#1f77b4','#d62728','#2ca02c','#9467bd','#8c564b','#e377c2','#7f7f7f','#ff7f0e','#bcbd22','#17becf','#f8e520'],
                 showColLabel= ["","X","Y","Y2"],
                 showColLabelUnit=["","X","Y","Y2"],
                 fill="_",
                 show=None,
                 filename=None,
                 labels=None,
                 colorOffset=0,
                 errors=None,
                 showErrorOnlyEvery=1,
                 erroralpha=0.1,
                 ax2erroralpha=0.3,
                 erroralphabar=0.2,
                 ax2erroralphabar=0.3,
                 capsize=2,
                 capthick=1,
                 errorTypeUp=1,
                 errorTypeDown=1,
                 ax2errorTypeUp=1,
                 ax2errorTypeDown=1,
                 ls="-",
                 ax2ls="--",
                 ax2LegendLabelAddString="",
                 overrideErrorTypes=False,
                 overrideFileList=False,
                 dataProcessed=False,
                 averageProcessed=False,
                 dataImported=False,
                 dataList=None,
                 errList=[None]*3
                ):
        #static inits
        self.ax=None
        self.ax2=None
        #dyn inits
        self.fileFormat=fileFormat
        if len(showColTup) is not 2:
            raise
        self.showColTup=showColTup
        self.showCol=self.showColTup[0] 
        self.showCol2=self.showColTup[1]
        self.fill=fill
        self.showColAxType=showColAxType
        self.showColAxLim=showColAxLim
        self.xCol=xCol
        self.colors=colors
        self.showColLabel=showColLabel
        self.showColLabelUnit=showColLabelUnit
        if xLim is not None:
            if len(xLim) is 2:
                self.showColAxLim[xCol]=xLim  
        if yLim is not None:
            if len(yLim) is 2:
                try:
                    if len(yLim[0]) is 2:
                        self.showColAxLim[self.showCol]=yLim[0]
                        self.showColAxLim[self.showCol2]=yLim[1]
                    else:
                        raise
                except TypeError:
                    self.showColAxLim[self.showCol]=yLim
        self.scaleX=scaleX
        if scaleX < 0.6 and customFontsize is None:
            self.customFontsize=[10,10,6,6,6]
        elif scaleX >= 0.6 and customFontsize is None:
            self.customFontsize=None
        else:
            self.customFontsize=customFontsize
        self.averageMedian=averageMedian
        self.xColOrig=xColOrig
        self.xLimOrig=xLimOrig
        self.xLim=self.showColAxLim[xCol]
        self.axYLim=self.showColAxLim[self.showCol]
        self.ax2YLim=self.showColAxLim[self.showCol2]
        self.xLabel=self.showColLabelUnit[xCol]
        self.ax2Labels=ax2Labels
        self.axYLabel=self.showColLabelUnit[self.showCol]
        self.ax2YLabel=self.showColLabelUnit[self.showCol2]
        self.ax2LegendLabelAddString=ax2LegendLabelAddString
        if title is None:
            self.title="Plot of "+name
        else:
            self.tilte=title
        self.name=name
        self.HWratio=HWratio
        self.titleBool=titleBool
        self.fig_width_pt=fig_width_pt
        self.legendEdgeSize=legendEdgeSize*scaleX
        self.colorOffset=colorOffset
        self.showErrorOnlyEvery=showErrorOnlyEvery
        self.erroralpha=erroralpha
        self.ax2erroralpha=ax2erroralpha
        self.erroralphabar=erroralphabar
        self.ax2erroralphabar=ax2erroralphabar
        self.capsize=capsize*self.scaleX
        self.capthick=capthick*self.scaleX
        self.errorTypeUp=errorTypeUp
        self.errorTypeDown=errorTypeDown
        self.ax2errorTypeUp=ax2errorTypeUp
        self.ax2errorTypeDown=ax2errorTypeDown
        self.ls=ls
        self.ax2ls=ax2ls
        self.overrideErrorTypes=overrideErrorTypes
        self.overrideFileList=overrideFileList
        self.filename=filename
        self.dataProcessed=dataProcessed
        self.averageProcessed=averageProcessed
        self.dataImported=dataImported
        self.dataList=dataList
        self.expectData=errList[0]
        self.deviaData=errList[1]
        self.logErr=errList[2]
        #inits
        self.__initFileList(fileList, errors, labels, show)
        self.dataList=self.__importData()
        
    def _figsize(self):
        inches_per_pt = 1.0/72.27                       # Convert pt to inch
        fig_width = self.fig_width_pt*inches_per_pt*self.scaleX   # width in inches
        fig_height = fig_width*self.HWratio                  # height in inches
        fig_size = [fig_width,fig_height]
        return fig_size

    def _newFig(self):
        matplotlib.pyplot.clf()
        self.__initTex(customFontsize=self.customFontsize)
        fig = matplotlib.pyplot.figure(figsize=self._figsize())
        ax = fig.add_subplot(111)
        return fig, ax

    def __initTex(self, customFontsize=None):
        #self._resetRcParams()
        if self.customFontsize is not None and len(self.customFontsize) is 5:
            pgf_with_pdflatex = {
                "pgf.texsystem": pgfSys,
                "font.family": "serif", # use serif/main font for text elements
                "font.size": self.customFontsize[0],
                "axes.labelsize": self.customFontsize[1],               # LaTeX default is 10pt font.
                "legend.fontsize": self.customFontsize[2],               # Make the legend/label fonts a little smaller
                "xtick.labelsize": self.customFontsize[3],
                "ytick.labelsize": self.customFontsize[4],
                "text.usetex": True,    # use inline math for ticks
                "pgf.rcfonts": False, 
                "pgf.preamble": texPreamble
            }
        else:
            pgf_with_pdflatex = {
                "pgf.texsystem": pgfSys,
                "font.family": "serif", # use serif/main font for text elements
                "font.size": 12,
                "axes.labelsize": "medium",               # LaTeX default is 10pt font.
                "legend.fontsize": "small",               # Make the legend/label fonts a little smaller
                "xtick.labelsize": "medium",
                "ytick.labelsize": "medium",
                "text.usetex": True,    # use inline math for ticks
                "pgf.rcfonts": False, 
                "pgf.preamble": texPreamble
            }
        mpl.rcParams.update(pgf_with_pdflatex)
        self._scaleRcParams()
        
     
    def _scaleRcParams(self):
        mpl.rcParams["lines.linewidth"]=self.scaleX*mpl.rcParams["lines.linewidth"]
        mpl.rcParams['axes.linewidth'] = self.scaleX*mpl.rcParams["axes.linewidth"]
        mpl.rcParams['xtick.major.size'] = self.scaleX*mpl.rcParams['xtick.major.size']
        mpl.rcParams['xtick.major.width'] = self.scaleX*mpl.rcParams['xtick.major.width']
        mpl.rcParams['xtick.minor.size'] = self.scaleX*mpl.rcParams['xtick.minor.size']
        mpl.rcParams['xtick.minor.width'] = self.scaleX*mpl.rcParams['xtick.minor.width']
        mpl.rcParams['ytick.major.size'] = self.scaleX*mpl.rcParams['ytick.major.size']
        mpl.rcParams['ytick.major.width'] = self.scaleX*mpl.rcParams['ytick.major.width']
        mpl.rcParams['ytick.minor.size'] = self.scaleX*mpl.rcParams['ytick.minor.size']
        mpl.rcParams['ytick.minor.width'] = self.scaleX*mpl.rcParams['ytick.minor.width']
        mpl.rcParams['grid.linewidth'] = self.scaleX*mpl.rcParams['grid.linewidth']
        
        
    def _rescaleRcParams(self):
        mpl.rcParams["lines.linewidth"]=1/self.scaleX*mpl.rcParams["lines.linewidth"]
        mpl.rcParams['axes.linewidth'] = 1/self.scaleX*mpl.rcParams["axes.linewidth"]
        mpl.rcParams['xtick.major.size'] = 1/self.scaleX*mpl.rcParams['xtick.major.size']
        mpl.rcParams['xtick.major.width'] = 1/self.scaleX*mpl.rcParams['xtick.major.width']
        mpl.rcParams['xtick.minor.size'] = 1/self.scaleX*mpl.rcParams['xtick.minor.size']
        mpl.rcParams['xtick.minor.width'] = 1/self.scaleX*mpl.rcParams['xtick.minor.width']
        mpl.rcParams['ytick.major.size'] = 1/self.scaleX*mpl.rcParams['ytick.major.size']
        mpl.rcParams['ytick.major.width'] = 1/self.scaleX*mpl.rcParams['ytick.major.width']
        mpl.rcParams['ytick.minor.size'] = 1/self.scaleX*mpl.rcParams['ytick.minor.size']
        mpl.rcParams['ytick.minor.width'] = 1/self.scaleX*mpl.rcParams['ytick.minor.width']
        mpl.rcParams['grid.linewidth'] = 1/self.scaleX*mpl.rcParams['grid.linewidth']
        
    
    
    
    def saveFig(self):
        matplotlib.pyplot.savefig(self.processFileName(option=".pdf"))
        matplotlib.pyplot.savefig(self.processFileName(option=".pgf"))
        
    def processFileName(self, option=".pdf"):
        if self.filename is None:
            if self.showCol2 == 0:
                string=self.name.replace(" ","")+self.fill+self.showColLabel[self.showCol].replace(" ","")
            else:
                string=self.name.replace(" ","")+self.fill+self.showColLabel[self.showCol].replace(" ","")+"+"+self.showColLabel[self.showCol2].replace(" ","")
        else:
            string=self.filename.replace(" ","")+self.fill+self.showColLabel[self.showCol].replace(" ","")
        if not self.scaleX is 1:
            string+=self.fill+"scaledWith{:03.0f}Pct".format(self.scaleX*100)
        return string+option
        
        
    def __importData(self):
        if not self.dataImported:
            self.dataList=[[Data(fileToNpArray(measurement, **self.fileFormat)[0], xCol=self.xCol, yCol=self.showCol) for measurement in sample] for sample in self.fileList]
        return self.dataList
    
    def processData(self):
        if not self.dataProcessed:
            for deviceData in self.dataList:
                for data in deviceData:
                    data.limitData(xLim=xLim)
            self.dataProcessed=True
        return self.dataList
    
    #
    #
    #   returns List with arithmetic averaged values, List with standarddeviation values for each sample
    @functools.lru_cache()
    def processAvg(self):
        dataList=self.dataList
        qtyCol=len(dataList[0][0].getData()[0])
        dataColList=[[[data.getData()[:,m] for data in deviceData] for deviceData in dataList] for m in range(0,qtyCol)]
        avgColList=[[np.average(element, axis=0) for element in column] for column in dataColList]
        avgList=[[avgColList[m][n] for m in range(0,qtyCol)] for n in range(0,len(dataList))]
        avgDataList=[Data.mergeData(avgData) for avgData in avgList]
        devDataList=[np.sqrt(np.sum([np.square(np.subtract(data,tempData.getData())) for tempData in deviceData],axis=0)/len(deviceData)) for data, deviceData in zip(avgDataList, dataList)]
        return avgDataList, devDataList
        
    #
    #
    #   returns List with median averaged values, List with standarddeviation values for each sample
    @functools.lru_cache()
    def processMedian(self):
        dataList=self.dataList
        qtyCol=len(dataList[0][0].getData()[0])
        dataColList=[[[data.getData()[:,m] for data in deviceData] for deviceData in dataList] for m in range(0,qtyCol)]
        medColList=[[np.median(element, axis=0) for element in column] for column in dataColList]
        medList=[[medColList[m][n] for m in range(0,qtyCol)] for n in range(0,len(dataList))]
        medDataList=[Data.mergeData(medData) for medData in medList]
        devDataList=[np.sqrt(np.sum([np.square(np.subtract(data,tempData.getData())) for tempData in deviceData],axis=0)/len(deviceData)) for data, deviceData in zip(medDataList, dataList)]
        return medDataList, devDataList
    
    def calcLogErr(self, expectData, deviaData):
        if not self.overrideErrorTypes:
            if self.showColAxType[self.showCol] == "log":
                self.errorTypeUp=0
                self.errorTypeDown=0
            else:
                self.errorTypeUp=1
                self.errorTypeDown=1
            if self.showColAxType[self.showCol2] == "log":
                self.ax2errorTypeUp=0
                self.ax2errorTypeDown=0
            else:
                self.ax2errorTypeUp=1
                self.ax2errorTypeDown=1 
        minData=[np.amin([data.getData() for data in deviceData], axis=0) for deviceData in self.dataList]
        maxData=[np.amax([data.getData() for data in deviceData], axis=0) for deviceData in self.dataList]
        symErr=[(np.absolute(expect.getData()-devia.getData())) for devia,expect in zip(deviaData, expectData)]
        minErr=[np.absolute(expect.getData()-mind) for devia,expect,mind in zip(deviaData, expectData, minData)]
        maxErr=[np.absolute(expect.getData()-maxi) for devia,expect,maxi in zip(deviaData, expectData, maxData)]
        logErrMin=[np.minimum(err, mind) for err,mind in zip(symErr,minErr)]
        logErrMax=[np.minimum(err, maxi) for err,maxi in zip(symErr,maxErr)]
        self.logErr=[logErrMin,logErrMax]
        return self.logErr
    
    def processAverage(self):
        if not self.averageProcessed:    
            if self.averageMedian:
                expectData, deviaData = self.processMedian()
            else:
                expectData, deviaData = self.processAvg()
            self.expectData=[Data(data) for data in expectData]
            self.deviaData=[Data(data) for data in deviaData]
            self.logErr=self.calcLogErr(self.expectData,self.deviaData)
            self.averageProcessed=True
        return self.expectData, self.deviaData
    
    
    
    def processPlot(self):
        expectData=self.processAverage()[0]
        colors=self.colors
        labels=self.labels
        showCol=self.showCol
        showCol2=self.showCol2
        colorOffset=self.colorOffset
        ax=self.ax
        ax2=self.ax2
        for n in range(0,len(expectData)):
            if self.show[n][0]:
                if self.errors[n][0]:
                    AX=ax.errorbar(*expectData[n].getSplitData2D(xCol=self.xCol, yCol=showCol), yerr=[self.logErr[self.errorTypeDown][n][:,showCol-1],self.logErr[self.errorTypeUp][n][:,showCol-1]], c=colors[n], capsize=self.capsize, capthick=self.capthick , ls=self.ls, label=labels[n], errorevery=self.showErrorOnlyEvery)
                else:
                    AX=ax.errorbar(*expectData[n].getSplitData2D(xCol=self.xCol, yCol=showCol), c=colors[n], ls=self.ls, label=labels[n])
                for a in AX[1]:
                    a.set_alpha(self.erroralphabar)
                for b in AX[2]:
                    b.set_alpha(self.erroralpha)
            if self.show[n][1] and showCol2 is not 0:
                if self.errors[n][1]:
                    AX2=ax2.errorbar(*expectData[n].getSplitData2D(xCol=self.xCol, yCol=showCol2), yerr=[self.logErr[self.ax2errorTypeDown][n][:,showCol2-1],self.logErr[self.ax2errorTypeUp][n][:,showCol2-1]], capsize=self.capsize, capthick=self.capthick, c=colors[n+colorOffset], ls=self.ax2ls, label=labels[n]+self.ax2LegendLabelAddString,  errorevery=self.showErrorOnlyEvery)
                else:
                    AX2=ax2.errorbar(*expectData[n].getSplitData2D(yCol=showCol2), c=colors[n+colorOffset], ls=self.ax2ls, label=labels[n]+self.ax2LegendLabelAddString)
                for a in AX2[1]:
                    a.set_alpha(self.ax2erroralpha)
                for b in AX2[2]:
                    b.set_alpha(self.ax2erroralphabar)
        
    def doPlot(self):
        fig,self.ax = self._newFig()
        ax= self.ax
        ax.set_xlabel(self.xLabel)
        if self.showColAxType[self.xCol] == "log":
            ax.set_xscale("log", basex=10, subsy=[2,3,4,5,6,7,8,9])
        ax.set_ylabel(self.axYLabel)
        if self.showColAxType[self.showCol] == "log":
            ax.set_yscale("log", basex=10, subsy=[2,3,4,5,6,7,8,9])
        if self.axYLim is not None:
            ax.set_ylim(*self.axYLim)
        ax.grid(alpha=0.5, ls=":")
        if True in [a[1] for a in self.show] and self.showCol2 != 0:
            self.ax2= ax.twinx()
            ax2=self.ax2
            ax2.set_ylabel(self.ax2YLabel)
            if self.showColAxType[self.showCol2] == "log":
                ax2.set_yscale("log", basex=10, subsy=[2,3,4,5,6,7,8,9])
            if self.ax2YLim is not None:
                ax2.set_ylim(*self.ax2YLim)
        self.dataList=self.processData()
        self.expectData, self.deviaData=self.processAverage()
        self.processPlot()
        handles, labels=ax.get_legend_handles_labels()
        handles = [h[0] for h in handles]
        #labels = labels[0:self.devices]
        if True in [a[1] for a in self.show] and self.ax2Labels and self.showCol2 is not 0:
            handles2, labels2=ax2.get_legend_handles_labels()
            handles2 = [h[0] for h in handles2]
            #labels2 = labels2[0:self.devices]
            handles=handles+handles2
            labels=labels+labels2
        leg=ax.legend(handles, labels, loc=2, numpoints=1)
        leg.get_frame().set_linewidth(self.legendEdgeSize)
        if self.titleBool:
            ax.set_title(self.title, fontsize="x-large")
        matplotlib.pyplot.tight_layout()
        self.saveFig()
        self._rescaleRcParams()
        matplotlib.pyplot.close(fig)
        return [self,self.processFileName(option=".pdf")] #filename
    
    
    def doPlotName(self):
        fig,self.ax = self._newFig()
        ax= self.ax
        ax.set_xlabel(self.xLabel)
        if self.showColAxType[self.xCol] == "log":
            ax.set_xscale("log", basex=10, subsy=[2,3,4,5,6,7,8,9])
        ax.set_ylabel(self.axYLabel)
        if self.showColAxType[self.showCol] == "log":
            ax.set_yscale("log", basex=10, subsy=[2,3,4,5,6,7,8,9])
        if self.axYLim is not None:
            ax.set_ylim(*self.axYLim)
        ax.grid(alpha=0.5, ls=":")
        if True in [a[1] for a in self.show] and self.showCol2 != 0:
            self.ax2= ax.twinx()
            ax2=self.ax2
            ax2.set_ylabel(self.ax2YLabel)
            if self.showColAxType[self.showCol2] == "log":
                ax2.set_yscale("log", basex=10, subsy=[2,3,4,5,6,7,8,9])
            if self.ax2YLim is not None:
                ax2.set_ylim(*self.ax2YLim)
        self.dataList=self.processData()
        self.expectData, self.deviaData=self.processAverage()
        self.processPlot()
        handles, labels=ax.get_legend_handles_labels()
        handles = [h[0] for h in handles]
        #labels = labels[0:self.devices]
        if True in [a[1] for a in self.show] and self.ax2Labels and self.showCol2 is not 0:
            handles2, labels2=ax2.get_legend_handles_labels()
            handles2 = [h[0] for h in handles2]
            #labels2 = labels2[0:self.devices]
            handles=handles+handles2
            labels=labels+labels2
        leg=ax.legend(handles, labels, loc=2, numpoints=1)
        leg.get_frame().set_linewidth(self.legendEdgeSize)
        if self.titleBool:
            ax.set_title(self.title, fontsize="x-large")
        matplotlib.pyplot.tight_layout()
        self.saveFig()
        self._rescaleRcParams()
        matplotlib.pyplot.close(fig)
        return self.processFileName(option=".pdf")

class ListShapeException(Exception):
    pass    
   
    
        
                


