
# coding: utf-8

# In[1]:

pgfSys="lualatex"

import matplotlib as mpl
#mpl.use("pgf")
import numpy as np
import scipy.interpolate as inter
import matplotlib.pyplot
import copy
import sys
import os
import string
import warnings
import functools
import copy
import inspect
#import mpl_toolkits.axisartist as AA
from matplotlib import rc
from Filereader import fileToNpArray
from Data import Data
from Fitter import Fitter



# In[2]:


class Plot():
    
    #constants
    fig_width_default_pt=424.75906
    default_colors=[
        (31/255, 119/255, 180/255, 1),
        (214/255, 39/255, 40/255, 1),
        (148/255, 103/255, 189/255, 1),
        (140/255, 86/255, 75/255, 1),
        (227/255, 119/255, 194/255, 1),
        (127/255, 127/255, 127/255, 1),
        (255/255, 127/255, 14/255, 1),
        (188/255, 189/255, 34/255, 1),
        (23/255, 190/255, 207/255, 1),
        (248/255, 229/255, 32/255, 1),
        (44/255, 160/255, 44/255, 1)]
    
    
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
    @classmethod
    def create_line_string(cls, line,sep):
        string=""
        for element in line:
            string+=str(element)+sep
        return string[:-len(sep)]
    
    @classmethod
    def concentenate_files(cls, fileList, fileFormat={}, subdir="",  previous_subdir=""):
        try:
            end=fileFormat["fileEnding"]

        except KeyError:
            end=""
        try:    
            skip=fileFormat["skiplines"]
        except KeyError:
            skip=0
        con_filename=subdir+fileList[0][len(previous_subdir):]
        con_file=open(con_filename+end,"w")
        m=0
        for fname in fileList:
            with open(fname+end) as infile:
                n=0
                for line in infile:
                    if n>=skip or m==0:
                        con_file.write(line)
                    n+=1
                m+=1
        con_file.close()
        return con_filename
        
    
    def __initFileList(self, fileList, errors, labels, show, fitLabels, showLines, showMarkers):
        if self.overrideFileList:
            self.fileList=[]
            self.errors=[]
            self.labels=[]
            self.show=[]
            self.fitLabels=[]
        else:
            try:
                fileList_cor=[]
                for file in fileList:
                    fileList_sub_cor=[]
                    for filesub in file:
                        if isinstance(filesub,list):
                            new_file=self.concentenate_files(filesub)
                        else:
                            new_file=filesub
                        fileList_sub_cor.append(new_file)
                    fileList_cor.append(fileList_sub_cor)    
                self.fileList=fileList_cor
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
                self.errors=[[deviceLabel != "", deviceLabel != ""] for deviceLabel in self.labels]
                self.show=self.errors
            else:
                try:
                    if self.showCol == 0 or self.showCol2 == 0:
                        temp=[errors[m][0] for m in range(0,len(self.fileList))]
                    else:
                        temp=[errors[m][1] for m in range(0,len(self.fileList))]
                    self.errors=errors
                    self.show=self.errors
                except IndexError:
                    raise ListShapeException("The errors' list has to be formatted like: [[sampleAerrorAxis1,sampleAerrorAxis2],[sampleBerrorAxis1,sampleBerrorAxis2]]")
                except TypeError:
                    if type(errors) == bool or isinstance(errors, (list,tuple)):
                        self.show=[[True,True] for device in self.fileList]
                    else:
                        raise ListShapeException("The error's TypeError")
                
                self.labels=["Sample {:d}".format(m+1) for m in range(0,len(self.fileList))]
            if show is not None:
                try:
                    if self.showCol == 0 or self.showCol2 == 0:
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
            if fitLabels is None:
                self.fitLabels=["Fit of "+label for label in self.labels]
            else:
                self.fitLabels=fitLabels
            if isinstance(showLines,list):
                self.showLines=showLines
            else:
                self.showLines=[[showLines,showLines] for device in self.fileList]
            if isinstance(showMarkers,list):
                self.showMarkers=showMarkers
            else:
                self.showMarkers=[[showMarkers,showMarkers] for device in self.fileList]    

    def __init__(self,
                 name,
                 fileList,
                 fileFormat={"separator":"\t", "skiplines":1},
                 showColTup=(2,3),
                 xLim=None,
                 limCol=None,
                 xLimOrig=None,
                 yLim=None,
                 scaleX=1,
                 customFontsize=None,
                 averageMedian=False,
                 xCol=1,
                 xCol2=0,
                 xColOrig=1,
                 title=None,
                 HWratio=3/4, # height to width ratio
                 fig_width_pt=fig_width_default_pt, # get it by \the\textwidth
                 titleBool=True,
                 legendEdgeSize=1,
                 ax2Labels=True,
                 #spectralDataformat={"separator":";", "skiplines":75}, #jeti csv format
                 showColAxType=[None,"lin","lin","lin"],
                 xAxisLim=None,
                 showColAxLim=[None,None,None,None],
                 colors=default_colors,
                 linestyles=["-","--",":","-."],
                 markers=["o","^","s","p","P","*"],
                 iterLinestyles=False,
                 linestyleOffset=0,
                 iterMarkers=False,
                 #markerOffset=0,
                 showColLabel=None,
                 showColLabelUnitNoTex=[None,"X","Y","Y2"],
                 showColLabelUnit=[None,"X","Y","Y2"],
                 fill="_",
                 show=None,
                 filename=None,
                 labels=None,
                 colorOffset=0,
                 errors=None,
                 showErrorOnlyEvery=1,
                 erroralpha=0.5,
                 ax2erroralpha=0.5,
                 erroralphabar=0.5,
                 ax2erroralphabar=0.5,
                 capsize=2,
                 capthick=1,
                 errorTypeUp=1,
                 errorTypeDown=1,
                 ax2errorTypeUp=1,
                 ax2errorTypeDown=1,
                 ls="-",
                 mk="o",
                 ax2ls="--",
                 ax2mk="^",
                 ax2LegendLabelAddString="",
                 overrideErrorTypes=False,
                 overrideFileList=False,
                 dataProcessed=False,
                 averageProcessed=False,
                 dataImported=False,
                 dataList=None,
                 errList=[None]*3,
                 injCode="pass",
                 legLoc=0,
                 fitList=None,
                 fitLabels=None,
                 fitAlpha=0.75,
                 fitLs=":", 
                 fitColors=None,
                 showFitInLegend=True,
                 fixedFigWidth=False,
                 xAxis2=False,
                 xAxisTickLabels=None,
                 xAxisTicks=None,
                 legendBool=True,
                 titleFontsize="x-large",
                 customLabelAx2=None,
                 doNotFit=False,
                 font="Latin Modern Sans",
                 filenamePrefix=None,
                 newFileList=None,
                 concentenate_files_instead_of_avg=False,
                 no_plot=False,
                 useTex=True,
                 partialFitLabels=[],
                 showLines=True,
                 showMarkers=False,
                 markerSize=6.0,
                 markerFillstyles=['full','none'],
                 subdir=None,
                 iterBoth=False,
                 append_col_in_label=True,
                 #ax_aspect='auto',
                ):
        #static inits
        self.fig=None
        self.ax=None
        self.ax2=None
        self.axX2=None
        self.alreadyFitted=False
        #dyn inits
        self.fileFormat=fileFormat
        if len(showColTup) != 2:
            raise
        try:
            if showColTup[0]<=0 or showColTup[1]<0:
                raise
        except TypeError:
            pass
        self.showColTup=showColTup
        self.showCol=self.showColTup[0] 
        self.showCol2=self.showColTup[1]
        self.fill=fill
        self.showColAxType=showColAxType
        self.showColAxLim=showColAxLim
        self.xCol=xCol
        self.colors=colors
        if showColLabel is None:
            sCLU=copy.deepcopy(showColLabelUnit)
            try:
                sCLU[0]="?"
                self.showColLabel=[showColLabelUnitElement.split()[0] for showColLabelUnitElement in sCLU]
            except:
                raise
        else:
            self.showColLabel=showColLabel
        self.useTex=useTex
        if self.useTex:
            #mpl.use("pgf")
            self.showColLabelUnit=showColLabelUnit
        else:
            mpl.use("Qt5Agg")
            self.showColLabelUnit=showColLabelUnitNoTex
        self.showColLabelUnitNoTex=showColLabelUnitNoTex
        if yLim is not None:
            if len(yLim) == 2:
                try:
                    if len(yLim[0]) == 2:
                        self.showColAxLim[self.showCol]=yLim[0]
                        self.showColAxLim[self.showCol2]=yLim[1]
                    else:
                        Exception("yLim is of wrong format")
                except TypeError:
                    self.showColAxLim[self.showCol]=yLim
            else:
                raise Exception("yLim is of wrong format")
        
        self.scaleX=scaleX
        if scaleX < 0.6 and customFontsize is None:
            self.customFontsize=[10,10,6,6,6]
        elif scaleX >= 0.6 and customFontsize is None:
            self.customFontsize=(mpl.rcParams["font.size"],mpl.rcParams["axes.labelsize"],mpl.rcParams["legend.fontsize"],mpl.rcParams["xtick.labelsize"],mpl.rcParams["ytick.labelsize"])
        else:
            self.customFontsize=customFontsize
        self.averageMedian=averageMedian
        self.xColOrig=xColOrig
        self.xLimOrig=xLimOrig
        self.xLim=xLim
        self.axXLim=xAxisLim
        try:
            self.axYLim=self.showColAxLim[self.showCol]
            self.axYLabel=self.showColLabelUnit[self.showCol]
        except TypeError:
            self.axYLim=self.showColAxLim[self.showCol[0]]
            self.axYLabel=self.showColLabelUnit[self.showCol[0]]
        try:
            self.ax2YLim=self.showColAxLim[self.showCol2]
            self.ax2YLabel=self.showColLabelUnit[self.showCol2]
        except TypeError:
            self.ax2YLim=self.showColAxLim[self.showCol2[0]]
            self.ax2YLabel=self.showColLabelUnit[self.showCol2[0]]
        self.ax2Labels=ax2Labels
        
        
        #self.ax2LegendLabelAddString=ax2LegendLabelAddString 
        if title is None:
            self.title="Plot of "+name
        else:
            self.title=title
        self.name=name
        self.HWratio=HWratio
        self.titleBool=titleBool
        self.fig_width_pt=fig_width_pt
        self.legendEdgeSize=legendEdgeSize*scaleX
        self.colorOffset=colorOffset
        try:
            showErrorOnlyEvery[0]
            self.showErrorOnlyEvery=showErrorOnlyEvery
        except:       
            self.showErrorOnlyEvery=[showErrorOnlyEvery]*len(fileList)
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
        self.iterLinestyles=iterLinestyles
        self.linestyles=linestyles
        self.markers=markers
        self.linestyleOffset=linestyleOffset
        self.iterMarkers=iterMarkers
        #self.markerOffset=markerOffset
        if iterLinestyles:
            self.ls=linestyles[0]
            self.ax2ls=linestyles[0]
            self.ax1color=colors[0]
            self.ax2color=colors[1]
        else:
            self.ls=ls
            self.ax2ls=ax2ls
        if iterMarkers:    
            self.mk=markers[0]
            self.ax2mk=markers[0]
            self.ax1color=colors[0]
            self.ax2color=colors[1]
        else:
            self.mk=mk
            self.ax2mk=ax2mk
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
        self.injCode=injCode
        self.legLoc=legLoc
        self.fitList=fitList
        if fitColors is None:
            self.fitColors=self.colors
        else:
            self.fitColors =fitColors
        self.fitLs= fitLs
        self.fitAlpha= fitAlpha
        self.showFitInLegend=showFitInLegend
        self.fixedFigWidth=fixedFigWidth
        self.xAxis2=xAxis2
        self.xAxisTickLabels=xAxisTickLabels
        self.xAxisTicks=xAxisTicks
        self.xCol2=xCol2
        self.ax2XLabel=self.showColLabelUnit[self.xCol2]
        self.ax2XLim=self.showColAxLim[self.xCol2]
        self.legendBool=legendBool
        self.titleFontsize=titleFontsize
        self.limCol=limCol
        self.customLabelAx2=customLabelAx2
        self.doNotFit=doNotFit
        self.font=font
        self.filenamePrefix=filenamePrefix
        self.concentenate_files_instead_of_avg=concentenate_files_instead_of_avg
        self.no_plot=no_plot
        self.partialFitLabels=partialFitLabels
        self.markerSize=markerSize
        self.markerFillstyles=markerFillstyles
        self.iterBoth=iterBoth
        self.append_col_in_label=append_col_in_label
        #self.ax_aspect=ax_aspect
        #inits
        #if mpl_use == "pgf":
        #    self.mpl_tex = True;
        #else:
        #    self.mpl_tex = False
        if newFileList is not None:
            self.__initFileList(newFileList, errors, labels, show, fitLabels, showLines, showMarkers)
        else:
            self.__initFileList(fileList, errors, labels, show, fitLabels, showLines, showMarkers)
        self.dataList=self.importData()

        
    def _set_size(w,h, ax=None):
        """ w, h: width, height in inches """
        if not ax: ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)

        
    def _figsize(self):
        inches_per_pt = 1.0/72.27# Convert pt to inch
        if self.fixedFigWidth:
            fig_width = self.fig_width_pt*inches_per_pt
        else:
            fig_width = self.fig_width_pt*inches_per_pt*self.scaleX   # width in inches
        fig_height = fig_width*self.HWratio                  # height in inches
        fig_size = [fig_width,fig_height]
        return fig_size

    def _newFig(self):
        matplotlib.pyplot.clf()
        self.__initTex(customFontsize=self.customFontsize)
        fig = matplotlib.pyplot.figure(figsize=self._figsize())
        ax = fig.add_subplot(111)
        #ax.set_aspect(self.ax_aspect)
        #ax=AA.Subplot(fig, 111)
        #fig.add_subplot(ax)
        return fig, ax

    def __initTex(self, customFontsize=None):
        #self._resetRcParams()
        pgf_with_lualatex={
                "pgf.texsystem": pgfSys,
                "font.family": "sans-serif", # use serif/main font for text elements
                "font.sans-serif": [self.font],
                "mathtext.fallback": "cm",
                "axes.unicode_minus": False,
                "mathtext.fontset":"custom",
                "mathtext.tt": self.font,
                "mathtext.rm": self.font,
                "mathtext.sf": self.font,
                "mathtext.it": self.font,
                "font.size": self.customFontsize[0],
                "axes.labelsize": self.customFontsize[1],               # LaTeX default is 10pt font.
                "legend.fontsize": self.customFontsize[2],               # Make the legend/label fonts a little smaller
                "xtick.labelsize": self.customFontsize[3],
                "ytick.labelsize": self.customFontsize[4],
                "text.usetex": True,    # use inline math for ticks
                "axes.formatter.use_mathtext": True,
                "pgf.rcfonts": False, 
                "pgf.preamble": r"\usepackage{amsmath}\usepackage{upgreek}\usepackage{lmodern}\usepackage{sfmath}",#"#\usepackage{lmodern}\usepackage{fontspec}\setmainfont{Latin Modern Sans}",
                #"text.latex.preamble": r"\usepackage{amsmath}\usepackage{upgreek}\usepackage{lmodern}\usepackage{sfmath}\setmainfont{Latin Modern Sans}",#"\usepackage{lmodern}\usepackage{fontspec}\setmainfont{Latin Modern Sans}",
                "lines.markersize": self.markerSize
            }
        if self.customFontsize is not None and len(self.customFontsize) == 5:
            pgf_with_lualatex.update({
                "font.size": self.customFontsize[0],
                "axes.labelsize": self.customFontsize[1],               # LaTeX default is 10pt font.
                "legend.fontsize": self.customFontsize[2],               # Make the legend/label fonts a little smaller
                "xtick.labelsize": self.customFontsize[3],
                "ytick.labelsize": self.customFontsize[4],
            })
        else:
            pgf_with_lualatex.update({
                "pgf.texsystem": pgfSys,
                "font.size": 12,
                "axes.labelsize": "medium",               # LaTeX default is 10pt font.
                "legend.fontsize": "small",               # Make the legend/label fonts a little smaller
                "xtick.labelsize": "medium",
                "ytick.labelsize": "medium",
            })
        if self.useTex== True:
            mpl.rcParams.update(pgf_with_lualatex)
        else:
            mpl.rcParams.update({"text.usetex": False, "backend":"Qt5Agg"})
        self._scaleRcParams()
    
    #fitTuple ([start, end], [show_start, show_end],func , (param1,param2), (textXPos,textYPos), desc, addKwArgs)
    def __initFitter(self):
        fitterList=[]
        for expect, devia, fitTuple in zip(self.expectData,self.deviaData,self.fitList):
            if fitTuple == () or fitTuple is None:
                fitterList.append(None)
            elif type(fitTuple) is not tuple and type(fitTuple) is list:
                fitSubList=[]
                n=0
                for fTuple in fitTuple:
                    if n==0:
                        expectCopy=expect
                        deviaCopy=devia
                    else:
                        expectCopy=copy.deepcopy(expect)
                        deviaCopy=copy.deepcopy(devia)
                    n+=1
                    fitSubList.append(Fitter(expectCopy,
                       fTuple[2],
                       errorData=deviaCopy,
                       dataForFitXLim=fTuple[0],
                       dataForFitYLim=fTuple[6].pop("dataForFitYLim",None),
                       curveDataXLim=fTuple[1],
                       params=fTuple[3],
                       textPos=fTuple[4],
                       desc=fTuple[5],
                       addKwArgs=fTuple[6]
                       ))
                fitterList.append(fitSubList)

            else:
                fitterList.append(Fitter(expect,
                       fitTuple[2],
                       errorData=devia,
                       dataForFitXLim=fitTuple[0],
                       dataForFitYLim=fitTuple[6].pop("dataForFitYLim",None),
                       curveDataXLim=fitTuple[1],
                       params=fitTuple[3],
                       textPos=fitTuple[4],
                       desc=fitTuple[5],
                       addKwArgs=fitTuple[6]
                       ))
        return fitterList
    
    def limit_fit_data_and_fit(self, fitter):
        if isinstance(fitter.dataForFitXLim[0],list):
            feature=[11,2]
        else:
            feature=[1,2]
        if isinstance(fitter.curveDataXLim[0],list):
            feature[1]=21
        else:
            feature[1]=2
        fitter.limitData(xLim=fitter.dataForFitXLim, yLim=fitter.dataForFitYLim, feature=feature[0])
        if not self.doNotFit:
            try:
                fitter.fit(xCol=self.xCol, yCol=self.showCol, p0=fitter.params)
            except RuntimeError as err:
                raise FitException(self.fitterList.index(fitter),self.fitList[(self.fitterList.index(fitter))],err,fitter.params)
        fitter.doFitCurveData(xCol=self.xCol)
        fitter.limitData(xLim=fitter.curveDataXLim, feature=feature[1])
    
    def __processFit(self):
        for fitter in self.fitterList:
            if fitter is not None:
                if type(fitter) is list:
                    for subFitter in fitter: 
                        self.limit_fit_data_and_fit(subFitter)
                else:
                    self.limit_fit_data_and_fit(fitter)
                    
            
            
            
     
    def _scaleRcParams(self):
        mpl.rcParams["lines.linewidth"]=self.scaleX*mpl.rcParams["lines.linewidth"]
        mpl.rcParams["lines.markeredgewidth"]=self.scaleX*mpl.rcParams["lines.markeredgewidth"]
        mpl.rcParams["lines.markersize"]=self.scaleX*mpl.rcParams["lines.markersize"]
        mpl.rcParams['axes.linewidth'] = self.scaleX*mpl.rcParams["axes.linewidth"]
        mpl.rcParams['xtick.major.size'] = self.scaleX*mpl.rcParams['xtick.major.size']
        mpl.rcParams['xtick.major.width'] = self.scaleX*mpl.rcParams['xtick.major.width']
        mpl.rcParams['xtick.major.pad'] = self.scaleX*mpl.rcParams['xtick.major.pad']
        mpl.rcParams['xtick.minor.size'] = self.scaleX*mpl.rcParams['xtick.minor.size']
        mpl.rcParams['xtick.minor.width'] = self.scaleX*mpl.rcParams['xtick.minor.width']
        mpl.rcParams['xtick.minor.pad'] = self.scaleX*mpl.rcParams['xtick.minor.pad']
        mpl.rcParams['ytick.major.size'] = self.scaleX*mpl.rcParams['ytick.major.size']
        mpl.rcParams['ytick.major.width'] = self.scaleX*mpl.rcParams['ytick.major.width']
        mpl.rcParams['ytick.major.pad'] = self.scaleX*mpl.rcParams['ytick.major.pad']
        mpl.rcParams['ytick.minor.size'] = self.scaleX*mpl.rcParams['ytick.minor.size']
        mpl.rcParams['ytick.minor.width'] = self.scaleX*mpl.rcParams['ytick.minor.width']
        mpl.rcParams['ytick.minor.pad'] = self.scaleX*mpl.rcParams['ytick.minor.pad']
        mpl.rcParams['grid.linewidth'] = self.scaleX*mpl.rcParams['grid.linewidth']
        
        
    def _rescaleRcParams(self):
        mpl.rcParams["lines.linewidth"]=1/self.scaleX*mpl.rcParams["lines.linewidth"]
        mpl.rcParams["lines.markeredgewidth"]=1/self.scaleX*mpl.rcParams["lines.markeredgewidth"]
        mpl.rcParams["lines.markersize"]=1/self.scaleX*mpl.rcParams["lines.markersize"]
        mpl.rcParams['axes.linewidth'] = 1/self.scaleX*mpl.rcParams["axes.linewidth"]
        mpl.rcParams['xtick.major.size'] = 1/self.scaleX*mpl.rcParams['xtick.major.size']
        mpl.rcParams['xtick.major.width'] = 1/self.scaleX*mpl.rcParams['xtick.major.width']
        mpl.rcParams['xtick.major.pad'] = 1/self.scaleX*mpl.rcParams['xtick.major.pad']
        mpl.rcParams['xtick.minor.size'] = 1/self.scaleX*mpl.rcParams['xtick.minor.size']
        mpl.rcParams['xtick.minor.width'] = 1/self.scaleX*mpl.rcParams['xtick.minor.width']
        mpl.rcParams['xtick.minor.pad'] = 1/self.scaleX*mpl.rcParams['xtick.minor.pad']
        mpl.rcParams['ytick.major.size'] = 1/self.scaleX*mpl.rcParams['ytick.major.size']
        mpl.rcParams['ytick.major.width'] = 1/self.scaleX*mpl.rcParams['ytick.major.width']
        mpl.rcParams['ytick.major.pad'] = 1/self.scaleX*mpl.rcParams['ytick.major.pad']
        mpl.rcParams['ytick.minor.size'] = 1/self.scaleX*mpl.rcParams['ytick.minor.size']
        mpl.rcParams['ytick.minor.width'] = 1/self.scaleX*mpl.rcParams['ytick.minor.width']
        mpl.rcParams['ytick.minor.pad'] = 1/self.scaleX*mpl.rcParams['ytick.minor.pad']
        mpl.rcParams['grid.linewidth'] = 1/self.scaleX*mpl.rcParams['grid.linewidth']
        
    
    
    
    def saveFig(self):
        try:
            if self.useTex:
                self.fig.savefig(self.processFileName(option=".pdf"), bbox_inches='tight')
                self.fig.savefig(self.processFileName(option=".pgf"), bbox_inches='tight')
            else:
                #matplotlib.pyplot.show()
                self.fig.savefig(self.processFileName(option=".png"))
        except ValueError:
            if self.useTex:
                self.fig.savefig(self.processFileName(option=".pdf"))
                self.fig.savefig(self.processFileName(option=".pgf"))
            else:
                #matplotlib.pyplot.show()
                self.fig.savefig(self.processFileName(option=".png"))
            
    def processFileName_makedirs(self):
        try:
            folder=os.path.dirname(self.filenamePrefix)
            os.makedirs(folder)
        except TypeError:
            pass
        except FileExistsError as e:
            if e.errno != 17:
                raise
        except FileNotFoundError as f:
            if f.errno!=2:
                raise
        
    def processFileName(self, option=".pdf"):
        string=""
        if self.filename is None:
            if self.showCol2 == 0:
                string+=self.name.replace(" ","")+self.fill+self.showColLabel[self.showCol].replace(" ","")
            else:
                string+=self.name.replace(" ","")+self.fill+self.showColLabel[self.showCol].replace(" ","")+"+"+self.showColLabel[self.showCol2].replace(" ","")
        else:
            string=self.filename.replace(" ","")+self.fill+self.showColLabel[self.showCol].replace(" ","")
        if self.scaleX != 1:
            string+=self.fill+"scaledWith{:03.0f}Pct".format(self.scaleX*100)
        if self.filenamePrefix is not None:
            self.processFileName_makedirs()
            string=self.filenamePrefix+self.fill+string
        return string+option
        
    def makeDataFromFile(self, measurement, fileFormat, lower_crop=0, upper_crop=0):
        if upper_crop==0:
            return Data(
                        fileToNpArray(measurement, **fileFormat)[0][lower_crop:], 
                        desc=fileToNpArray(measurement, 
                                           **fileFormat)[1], 
                        xCol=self.xCol, 
                        yCol=self.showCol)
        return Data(
                    fileToNpArray(measurement, **fileFormat)[0][lower_crop:upper_crop], 
                    desc=fileToNpArray(measurement, 
                                       **fileFormat)[1], 
                    xCol=self.xCol, 
                    yCol=self.showCol)
        
    def importData(self, **kwargs):
        if not self.dataImported:
            try:
                self.fileFormat[0]
                self.dataList=[[self.makeDataFromFile(measurement, fileFormat, **kwargs) for measurement in sample] for sample,fileFormat in zip(self.fileList,self.fileFormat)]
            except KeyError:
                self.dataList=[[self.makeDataFromFile(measurement, self.fileFormat, **kwargs) for measurement in sample] for sample in self.fileList]
        return self.dataList
    
    def processData(self):
        if not self.dataProcessed:
            for deviceData in self.dataList:
                for data in deviceData:
                    data.limitData(xLim=self.xLimOrig)
                    #print("limiting data to: "+str(self.xLimOrig))
            self.dataProcessed=True
        return self.dataList
    
    #
    #
    #   returns List with arithmetic averaged values, List with standarddeviation values for each sample
    @functools.lru_cache()
    def processAvg(self, dataList=None):
        if dataList is None:
            dataList=self.dataList
        errorData=True
        for data in dataList:
            try:
                qtyCol=len(dataList[0][0].getData()[0])
                errorData=False
            except IndexError:
                pass

        if not errorData:
            qtyCol=len(dataList[0][0].getData()[0])
            dataColList=[[[data.getData()[:,m] for data in deviceData] for deviceData in dataList] for m in range(0,qtyCol)]
            descList=[deviceData[0].desc for deviceData in dataList]
            avgColList=[[np.average(element, axis=0) for element in column] for column in dataColList]
            avgList=[[avgColList[m][n] for m in range(0,qtyCol)] for n in range(0,len(dataList))]
            avgDataList=[Data.mergeData(avgData) for avgData in avgList]
            devDataList=[np.sqrt(np.sum([np.square(np.subtract(data,tempData.getData())) for tempData in deviceData],axis=0)/len(deviceData)) for data, deviceData in zip(avgDataList, dataList)]
            avgData=[Data(data, desc=desc, xCol=self.xCol, yCol=self.showCol) for data,desc in zip(avgDataList,descList)]
            devData=[Data(data, xCol=self.xCol, yCol=self.showCol) for data in devDataList]
        else:
            avgData=[]
            devData=[]
        return avgData, devData

    #
    #
    #   returns List with median averaged values, List with standarddeviation values for each sample
    @functools.lru_cache()
    def processMedian(self, dataList=None):
        if dataList is None:
            dataList=self.dataList
        errorData=True
        for data in dataList:
            try:
                qtyCol=len(data[0].getData()[0])
                errorData=False
            except IndexError:
                pass
        if not errorData:
            dataColList=[[[data.getData()[:,m] for data in deviceData] for deviceData in dataList] for m in range(0,qtyCol)]
            descList=[]
            for deviceData in dataList:
                try:
                    descList.append(deviceData[0].desc)
                except IndexError:
                    descList.append("")
            medColList=[[np.median(element, axis=0) for element in column] for column in dataColList]
            medList=[[medColList[m][n] for m in range(0,qtyCol)] for n in range(0,len(dataList))]
            medDataList=[Data.mergeData(medData) for medData in medList]
            devDataList=[np.sqrt(np.sum([np.square(np.subtract(data,tempData.getData())) for tempData in deviceData],axis=0)/len(deviceData)) for data, deviceData in zip(medDataList, dataList)]
            medData=[Data(data, desc=desc, xCol=self.xCol, yCol=self.showCol) for data,desc in zip(medDataList,descList)]
            devData=[Data(data, xCol=self.xCol, yCol=self.showCol) for data in devDataList]
        else:
            medData=[]
            devData=[]
        return medData, devData
    
    def processCustomMedian(self, dataList):
        qtyCol=len(dataList[0][0].getData()[0])
        dataColList=[[[data.getData()[:,m] for data in deviceData] for deviceData in dataList] for m in range(0,qtyCol)]
        medColList=[[np.median(element, axis=0) for element in column] for column in dataColList]
        medList=[[medColList[m][n] for m in range(0,qtyCol)] for n in range(0,len(dataList))]
        medDataList=[Data.mergeData(medData) for medData in medList]
        devDataList=[np.sqrt(np.sum([np.square(np.subtract(data,tempData.getData())) for tempData in deviceData],axis=0)/len(deviceData)) for data, deviceData in zip(medDataList, dataList)]
        return medDataList, devDataList
    
    def processCustomAvg(self, dataList):
        qtyCol=len(dataList[0][0].getData()[0])
        dataColList=[[[data.getData()[:,m] for data in deviceData] for deviceData in dataList] for m in range(0,qtyCol)]
        avgColList=[[np.average(element, axis=0) for element in column] for column in dataColList]
        avgList=[[avgColList[m][n] for m in range(0,qtyCol)] for n in range(0,len(dataList))]
        avgDataList=[Data.mergeData(avgData) for avgData in avgList]
        devDataList=[np.sqrt(np.sum([np.square(np.subtract(data,tempData.getData())) for tempData in deviceData],axis=0)/len(deviceData)) for data, deviceData in zip(avgDataList, dataList)]
        return avgDataList, devDataList
    
    #@functools.lru_cache()
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
        try:
            minData=[np.amin([data.getData() for data in deviceData], axis=0) for deviceData in self.dataList]
            maxData=[np.amax([data.getData() for data in deviceData], axis=0) for deviceData in self.dataList]
            symErr=[(np.absolute(expect.getData()-devia.getData())) for devia,expect in zip(deviaData, expectData)]
            minErr=[np.absolute(expect.getData()-mind) for devia,expect,mind in zip(deviaData, expectData, minData)]
            maxErr=[np.absolute(expect.getData()-maxi) for devia,expect,maxi in zip(deviaData, expectData, maxData)]
            logErrMin=[np.nan_to_num(np.minimum(err, mind), nan=0, posinf=0, neginf=0) for err,mind in zip(symErr,minErr)]
            logErrMax=[np.nan_to_num(np.minimum(err, maxi), nan=0, posinf=0, neginf=0) for err,maxi in zip(symErr,maxErr)]
            # if any error is as high that the bar would reach zero, set it to zero
            logErrMin=np.asarray([np.asarray([np.asarray([errv if value>errv else 0.0 for errv,value in zip(errv_row,value_row)], dtype=object) for errv_row,value_row in zip(err,expect.getData())], dtype=object) for err,expect in zip(logErrMin, expectData)], dtype=object)
            values=np.asarray([np.asarray([np.asarray([value for errv,value in zip(errv_row,value_row)], dtype=object) for errv_row,value_row in zip(err,expect.getData())], dtype=object) for err,expect in zip(logErrMin, expectData)], dtype=object)
            errvs=np.asarray([np.asarray([np.asarray([errv for errv,value in zip(errv_row,value_row)], dtype=object) for errv_row,value_row in zip(err,expect.getData())], dtype=object) for err,expect in zip(logErrMin, expectData)], dtype=object)
        except ValueError:
            warnings.warn("Unsupported Input for Error estimation given")
            logErrMin=[np.zeros(expect.getData().shape) for expect in expectData]
            logErrMax=logErrMin
        self.logErr=[logErrMin,logErrMax]
        return self.logErr
        
    @functools.lru_cache()
    def concentenate_data(self, dataList=None):
        if dataList is None:
            dataList=self.dataList
        new_dataList=[]
        new_devia_dataList=[]
        for dataSubList in dataList:
            dataArray=[]
            for data in dataSubList:
                dataArray.append(data.getData())
            try:
                device_data=Data(np.vstack(dataArray), xCol=self.xCol, yCol=self.showCol)
                device_data.removeDoubles()
                new_dataList.append(device_data)
                new_devia_dataList.append(Data(np.zeros(device_data.getData().shape), xCol=self.xCol, yCol=self.showCol))
            except ValueError:
                pass
        return new_dataList, new_devia_dataList
    
    def processAverage(self):
        if not self.averageProcessed:
            if not self.concentenate_files_instead_of_avg:
                try:
                    if self.averageMedian:
                        expectData, deviaData = self.processMedian()
                    else:
                        expectData, deviaData = self.processAvg()
                except (ValueError,IndexError):
                    raise
                    #warnings.warn("Unsupported Input for Error estimation given")
                    expectData, deviaData = ([],[])    
                self.expectData=expectData
                self.deviaData=deviaData
                self.logErr=self.calcLogErr(self.expectData,self.deviaData)
            else:
                self.expectData, self.deviaData=self.concentenate_data()
            #for expect,devia in zip(self.expectData,self.deviaData):
            #    expect.xCol=self.xCol
            #    devia.xCol=self.xCol
            self.averageProcessed=True
        return self.expectData, self.deviaData
    
    def calcCustomAverage(self, dataList):   
        if self.averageMedian:
            expectData,deviaData= self.processCustomMedian(dataList)
        else:
            expectData, deviaData = self.processCustomAvg(dataList)
        expectData=[Data(data) for data in expectData]
        deviaData=[Data(data) for data in deviaData]
        logErr=self.calcLogErr(expectData,deviaData)
        return expectData, logErr
    
    def xColTicksToXCol2Ticks(self, ticks):
        return ticks
    
    def afterPlot(self):
        return
    
    def processPlotSub(self, n, ls, mk, fs, ax2ls, ax2mk, ax2fs, ax1color, ax2color, data):
        colors=self.colors
        labels=self.labels
        xCol=self.xCol
        showCol=self.showCol
        showCol2=self.showCol2
        colorOffset=self.colorOffset
        ax=self.ax
        ax2=self.ax2
        linestyles=ls
        markerstyles=mk
        ax2linestyles=ax2ls
        ax2markerstyles=ax2mk
        expectData=data
        fillstyles=fs
        ax2fillstyles=ax2fs
        try:
            if self.append_col_in_label:
                labelY=labels[n]+" "+self.showColLabel[self.showCol]
            else:
                labelY=labels[n]
        except TypeError:
            if self.append_col_in_label:
                labelY=[labels[n]+" "+self.showColLabel[singleShowCol] for singleShowCol in self.showCol]
            else:
                labelY=labels[n]
        if self.show[n][0]:
            if self.errors[n][0]:
                try:
                    try:
                        yerr=[self.logErr[self.errorTypeDown][n][:,showCol-1],self.logErr[self.errorTypeUp][n][:,showCol-1]]
                    except IndexError:
                        yerr=None
                    AX=ax.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=showCol), yerr=yerr, c=ax1color[n], capsize=self.capsize, capthick=self.capthick , ls=linestyles[n], marker=markerstyles[n], label=labelY, errorevery=self.showErrorOnlyEvery[n], fillstyle=fillstyles[n])
                except TypeError:
                    for singleShowCol,singleLabelY in zip(showCol,labelY):
                        AX=ax.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=singleShowCol), yerr=[self.logErr[self.errorTypeDown][n][:,showCol-1],self.logErr[self.errorTypeUp][n][:,showCol-1]], c=ax1color[n], capsize=self.capsize, capthick=self.capthick, ls=linestyles[n], marker=markerstyles[n], label=singleLabelY, errorevery=self.showErrorOnlyEvery[n], fillstyle=fillstyles[n])
            else:
                try:
                    AX=ax.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=showCol), c=ax1color[n], ls=linestyles[n], marker=markerstyles[n], label=labelY, fillstyle=fillstyles[n])
                except TypeError:
                    for singleShowCol,singleLabelY in zip(showCol,labelY):
                        AX=ax.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=singleShowCol), c=ax1color[n], ls=linestyles[n], marker=markerstyles[n], label=singleLabelY, fillstyle=fillstyles[n])
            if self.fitList is not None and self.fitterList[n] is not None:
                if type(self.fitterList[n]) is list:
                    for fitter in self.fitterList[n]:
                        FIT=ax.errorbar(*fitter.CurveData.getSplitData2D(), c=self.fitColors[n], ls=self.fitLs, label=self.fitLabels[n], alpha=self.fitAlpha)
                else:
                    FIT=ax.errorbar(*self.fitterList[n].CurveData.getSplitData2D(), c=self.fitColors[n], ls=self.fitLs, label=self.fitLabels[n], alpha=self.fitAlpha)
            for a in AX[1]:
                a.set_alpha(self.erroralphabar)
            for b in AX[2]:
                b.set_alpha(self.erroralpha)
        if self.show[n][1] and self.showCol2 != 0:
            if self.customLabelAx2 is not None:
                if not isinstance(self.customLabelAx2, str):
                    labelZ=self.customLabelAx2[n]
                else:
                    labelZ=self.customLabelAx2
            else:
                try:
                    labelZ=labels[n]+" "+self.showColLabel[self.showCol2]
                except TypeError:
                    labelZ=[labels[n]+" "+self.showColLabel[singleShowCol2] for singleShowCol2 in showCol2]
            if self.errors[n][1]:
                try:
                    AX2=ax2.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=showCol2), yerr=[self.logErr[self.ax2errorTypeDown][n][:,showCol2-1],self.logErr[self.ax2errorTypeUp][n][:,showCol2-1]], capsize=self.capsize, capthick=self.capthick, c=ax2color[n], ls=ax2linestyles[n], marker=ax2markerstyles[n], label=labelZ,  errorevery=self.showErrorOnlyEvery[n], fillstyle=ax2fillstyles[n])
                except TypeError:
                    for singleShowCol2,singleLabelZ in zip(showCol2,labelZ):
                        AX2=ax2.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=singleShowCol2), yerr=[self.logErr[self.ax2errorTypeDown][n][:,showCol2-1],self.logErr[self.ax2errorTypeUp][n][:,showCol2-1]], capsize=self.capsize, capthick=self.capthick, c=ax2color[n], ls=ax2linestyles[n], marker=ax2markerstyles[n], label=singleLabelZ,  errorevery=self.showErrorOnlyEvery[n], fillstyle=ax2fillstyles[n])
            else:
                try:
                    AX2=ax2.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=showCol2), c=ax2color[n], ls=ax2linestyles[n], marker=ax2markerstyles[n], label=labelZ, fillstyle=ax2fillstyles[n])
                except TypeError:
                    for singleShowCol2,singleLabelZ in zip(showCol2,labelZ):
                        AX2=ax2.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=singleShowCol2), c=ax2color[n], ls=ax2linestyles[n], marker=ax2markerstyles[n], label=singleLabelZ, fillstyle=ax2fillstyles[n])
            if self.fitList is not None and self.fitterList[n] is not None:
                if type(self.fitterList[n]) is list:
                    for fitter in self.fitterList[n]:
                        try:
                            fitter.fit(yCol=self.showCol2)
                        except RuntimeError as err:
                            raise FitException(n,fitList[n],err)
                        fitter.doFitCurveData()
                        fitter.limitData(xLim=fitter.curveDataXLim, feature=2)
                        FIT2=ax2.errorbar(*fitter.CurveData.getSplitData2D(), c=self.fitColors[n+colorOffset], ls=self.fitLs, label=self.fitLabels[n], alpha=self.fitAlpha)
                else:
                    try:
                        self.fitterList[n].fit(yCol=self.showCol2)
                    except RuntimeError as err:
                        raise FitException(n,fitList[n],err)
                    self.fitterList[n].doFitCurveData()
                    self.fitterList[n].limitData(xLim=fitter.curveDataXLim, feature=2)
                    FIT2=ax2.errorbar(*self.fitterList[n].CurveData.getSplitData2D(), c=self.fitColors[n+colorOffset], ls=self.fitLs, label=self.fitLabels[n], alpha=self.fitAlpha)
            for a in AX2[1]:
                a.set_alpha(self.ax2erroralpha)
            for b in AX2[2]:
                b.set_alpha(self.ax2erroralphabar)
                
    def processPlot(self):
        expectData=self.expectData
        showMarkers=self.showMarkers
        showLines=self.showLines
        if self.iterLinestyles:
            linestyles=self.linestyles
            ax2linestyles=self.linestyles[::-self.linestyleOffset]
            ax1color=[self.ax1color]*len(expectData)
            ax2color=[self.ax2color]*len(expectData)
        else:
            if isinstance(self.ls, list):
                linestyles=self.ls
            else:
                linestyles=[self.ls]*len(expectData)
            ax2linestyles=[self.ax2ls]*len(expectData)
            ax1color=self.colors
            ax2color=self.colors
        if self.iterMarkers:
            markers=self.markers
            ax2markers=self.markers#[::-self.markerOffset]
            if self.iterBoth:
                ax1color=self.colors
                ax2color=self.colors
            else:
                ax1color=[self.ax1color]*len(expectData)
                ax2color=[self.ax2color]*len(expectData)
        else:
            if isinstance(self.mk, list):
                markers=self.mk
            else:
                markers=[self.mk]*len(expectData)
            ax2markers=[self.ax2mk]*len(expectData)
            ax1color=self.colors
            ax2color=self.colors
        fillstyles=[self.markerFillstyles[0]]*len(expectData)
        ax2fillstyles=[self.markerFillstyles[1]]*len(expectData)
        mk=[""]*len(markers)
        ls=[""]*len(linestyles)
        ax2mk=[""]*len(ax2markers)
        ax2ls=[""]*len(ax2linestyles)
        for n in range(0,len(expectData)):
            if showLines[n][0]:
                ls[n]=linestyles[n]
            if showMarkers[n][0]:
                mk[n]=markers[n]
            if showLines[n][1]:
                ax2ls[n]=ax2linestyles[n]
            if showMarkers[n][1]:
                ax2mk[n]=ax2markers[n]
            self.processPlotSub(n, ls, mk, fillstyles, ax2ls, ax2mk, ax2fillstyles, ax1color, ax2color, expectData)

    def doPlot(self):

        if not self.no_plot:
            self.fig, self.ax = self._newFig()
            ax= self.ax
            xLabel=self.showColLabelUnit[self.xCol]
            ax.set_xlabel(xLabel)
            if self.showColAxType[self.xCol] == "log":
                ax.set_xscale("log")#, basex=10, subsy=[2,3,4,5,6,7,8,9])
            ax.set_ylabel(self.axYLabel)
            #ax.set_xticklabels(ax.get_xticks())
            
            try:
                if self.showColAxType[self.showCol] == "log":
                    ax.set_yscale("log", nonpositive="clip")#, basex=10, subsy=[2,3,4,5,6,7,8,9])
            except TypeError:
                if self.showColAxType[self.showCol[0]] == "log":
                    ax.set_yscale("log", nonpositive="clip")#, basex=10, subsy=[2,3,4,5,6,7,8,9])
            if self.axYLim is not None:
                ax.set_ylim(*self.axYLim)
            if self.axXLim is not None:
                ax.set_xlim(*self.axXLim)
            ax.grid(True, alpha=0.5, linestyle=":")
            try:
                if self.showCol2 != 0:
                    if True in [a[1] for a in self.show]: 
                        self.ax2= ax.twinx()
                        ax2=self.ax2
                        ax2.set_ylabel(self.ax2YLabel)
                        if self.showColAxType[self.showCol2] == "log":
                            ax2.set_yscale("log")#, basex=10, subsy=[2,3,4,5,6,7,8,9])
                        if self.ax2YLim is not None:
                            ax2.set_ylim(*self.ax2YLim)
            except TypeError:
                if True in [a[1] for a in self.show]: 
                    self.ax2= ax.twinx()
                    ax2=self.ax2
                    ax2.set_ylabel(self.ax2YLabel)
                    if self.showColAxType[self.showCol2[0]] == "log":
                        ax2.set_yscale("log")#, basex=10, subsy=[2,3,4,5,6,7,8,9])
                    if self.ax2YLim is not None:
                        ax2.set_ylim(*self.ax2YLim)
                    #ax2.set_yticklabels(ax.get_yticks(), fontfamily=["DejaVuSans","sans-serif"])
            #ax.set_yticklabels(ax.get_yticks(), fontfamily=["DejaVuSans","sans-serif"], useTex=False)
            #ax.set_yticklabels(ax.get_yticks(), fontfamily=["DejaVuSans","sans-serif"])
            self.dataList=self.processData()
            self.expectData, self.deviaData=self.processAverage()
            if self.fitList is not None and not self.alreadyFitted:
                self.fitterList=self.__initFitter()
                self.__processFit()
                self.alreadyFitted=True
            self.processPlot()
            if self.xCol2 != 0:
                self.axX2=ax.twiny()
                axX2=self.axX2
                axX2.set_xlim(ax.get_xlim())
                axX2.set_xlabel(self.ax2XLabel)
                if self.showColAxType[self.xCol2] == "log":
                    axX2.set_xscale("log")#, basex=10, subsy=[2,3,4,5,6,7,8,9])
                if self.ax2XLim is not None:
                    ax2.set_xlim(*self.ax2XLim)
            #BEGIN: WARNING DANGEROUS!!!
            exec(self.injCode)
            #END: WARNING DANGEROUS!!!
            self.afterPlot()
            handles, labels=ax.get_legend_handles_labels()
            handles = [h[0] for h in handles]
            #labels = labels[0:self.devices]
            if True in [a[1] for a in self.show] and self.ax2Labels and self.showCol2 != 0:
                handles2, labels2=ax2.get_legend_handles_labels()
                handles2 = [h[0] for h in handles2]
                #labels2 = labels2[0:self.devices]
                handles=handles+handles2
                labels=labels+labels2
            if self.fitList is not None:
                index_list = [True if x not in self.fitLabels+self.partialFitLabels else False for x in labels]#
                orig_labels=copy.copy(labels)
                orig_handles=copy.copy(handles)
                labels = [l for l,index in zip(labels,index_list) if index]
                handles = [h for h,index in zip(handles,index_list) if index]
                if self.showFitInLegend:
                    labels += [l for l,index in zip(orig_labels,index_list) if not index]
                    handles += [h for h,index in zip(orig_handles,index_list) if not index]
            if self.legendBool:
                leg=ax.legend(handles, labels, loc=self.legLoc, numpoints=1)
                leg.get_frame().set_linewidth(self.legendEdgeSize)
            if self.titleBool:
                ax.set_title(self.title, fontsize=self.titleFontsize)
            matplotlib.pyplot.tight_layout()
            if self.xCol2 != 0:
                axX2.set_xticklabels(self.xColTicksToXCol2Ticks(ax.get_xticks()))
            self.saveFig()
            self._rescaleRcParams()
            matplotlib.pyplot.close(self.fig)
        return [self,self.processFileName(option=".pdf")] #filename
    
    def processAllAndExport(self, **kwargs):
        return self.exportAllData(**kwargs)
    
    def exportAllData(self, subdir="export/", fileEnd=".csv", colSep=",", fill=None, errorTypes=None, errorString="Error of ", expectData=None, errData=None, noError=False):
        filenames=[]
        if expectData is None:
            expectData=self.expectData
        if errData is None:
            errData=self.logErr 
        if fill is None:
            fill=self.fill
        if errorTypes is None:
            errorTypes=(0,)*(len(self.showColAxType)-1)
            for m in range(1,len(self.showColAxType)):
                if m == "log":
                    errorTypes[m-1]=1
        try:
            folder=os.path.dirname(subdir+self.name)
            os.makedirs(folder)
        except FileExistsError as e:
            if e.errno != 17:
                raise
        except FileNotFoundError as f:
            if f.errno!=2:
                raise
        for l in range(0,len(self.fileList)):
            filename=subdir+self.name.replace(" ","")+fill+self.labels[l].replace(" ","")+fileEnd
            filenames.append(filename)
            file = open(filename,"w")
            line = ""
            n=0
            for label in self.showColLabelUnitNoTex:
                if n != 0:
                    if noError:
                        line += label+colSep
                    else:
                        line += label+colSep+errorString+label+colSep
                n+=1
            line +="\n"
            file.write(line)
            expectArray=expectData[l].getData()
            for a in range(0,len(expectArray)): #a=row
                line=""
                for o in range(0,len(expectArray[a])): #o=element
                    if noError:
                        line+=str(expectArray[a][o])+colSep
                    else:
                        line+=str(expectArray[a][o])+colSep+str(errData[errorTypes[o]][l][a][o])+colSep
                line+="\n"
                file.write(line)
            file.close()
        return (self,filenames)
            
    
    def doPlotName(self):
        return self.doPlot()[1]
    
class ListShapeException(Exception):
    pass 

class FitException(Exception):
    def __init__(self,index, props, err, params):
        self.message="Error @ {} with Properties: {} \n Message: {} with parameters {}".format(index,props,str(err),str(params)) 
        