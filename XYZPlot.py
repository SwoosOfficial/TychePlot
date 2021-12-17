# coding: utf-8

import matplotlib as mpl
mpl.use("pgf")
import matplotlib.pyplot
import numpy as np
import pandas as pd
import copy
import sys
import os
import warnings
import copy
from Plot import Plot


class XYZPlot(Plot):

    def __init__(self,
                 name,
                 dataList=[],
                 fileList=[],
                 fileFormat={separator:"\t", skiplines:0, ignoreRowCol:(0,0)},
                 title=None,
                 axType=["lin","lin","lin"],
                 axLim=[None,None,None],
                 axLabel= ["X","Y", "Z",],
                 axLabelTex= None,
                 axLabelUnit=["X","Y", "Z",],
                 plotLine = None, # (ax,axval)
                ):
        
        self.fig = None
        self.dataList = dataList
        self.fileList = fileList
        self.title = title
        self.axType = axType
        self.axLim = axLim
        self.axLabel = axLabel
        self.axLabelTex = axLabelTex
        self.axLabelUnit = axLabelUnit
        self.plotLine = plotLine
        self.dataList = self.importData()
        
        
    def processFileName(self, option=".pdf"):
        if 
        appstring = self.fill+self.showColLabel[0].replace(" ","")+"+"+self.showColLabel[1].replace(" ","")+"_vs_"+self.showColLabel[2].replace(" ","")
        if self.filename is None:
            string=self.name.replace(" ","")+
        else:
            string=self.filename.replace(" ","")+self.fill+self.showColLabel[self.showCol].replace(" ","")
        if self.filenamePrefix is not None:
            string=self.filenamePrefix+self.fill+string
        return string+option
        
        
    def importData(self):
        if not self.dataImported:
            self.dataList=[[pd.read_csv for measurement in sample] for sample in self.fileList]
        return self.dataList
    
    def processData(self):
        if not self.dataProcessed:
            for deviceData in self.dataList:
                for data in deviceData:
                    data.limitData(xLim=self.xLimOrig)
                    #print("limiting data to: "+str(self.xLimOrig))
            self.dataProcessed=True
        return self.dataList
    
   
    def afterPlot(self):
        return
    
    def processPlot(self):
        

    def doPlot(self):
        if not self.no_plot:
            fig,self.ax = self._newFig()
            ax= self.ax
            xLabel=self.showColLabelUnit[self.xCol]
            ax.set_xlabel(xLabel)
            if self.showColAxType[self.xCol] == "log":
                ax.set_xscale("log")#, basex=10, subsy=[2,3,4,5,6,7,8,9])
            ax.set_ylabel(self.axYLabel)
            try:
                if self.showColAxType[self.showCol] == "log":
                    ax.set_yscale("log")#, basex=10, subsy=[2,3,4,5,6,7,8,9])
            except TypeError:
                if self.showColAxType[self.showCol[0]] == "log":
                    ax.set_yscale("log")#, basex=10, subsy=[2,3,4,5,6,7,8,9])
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
            self.dataList=self.processData()
            self.expectData, self.deviaData=self.processAverage()
            if self.fitList is not None:
                self.fitterList=self.__initFitter()
                self.__processFit()
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
            exec(self.injCode)
            self.afterPlot()
            handles, labels=ax.get_legend_handles_labels()
            handles = [h[0] for h in handles]
            #labels = labels[0:self.devices]
            if True in [a[1] for a in self.show] and self.ax2Labels and self.showCol2 is not 0:
                handles2, labels2=ax2.get_legend_handles_labels()
                handles2 = [h[0] for h in handles2]
                #labels2 = labels2[0:self.devices]
                handles=handles+handles2
                labels=labels+labels2
            if self.fitList is not None and not self.showFitInLegend:
                index_list = [True if x not in self.fitLabels else False for x in labels]
                labels = [l for l,index in zip(labels,index_list) if index]
                handles = [h for h,index in zip(handles,index_list) if index]
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
            matplotlib.pyplot.close(fig)
        return [self,self.processFileName(option=".pdf")] #filename
    
        