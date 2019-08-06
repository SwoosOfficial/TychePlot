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


class XYZPlot(Plot):

    def __init__(self,
                 name,
                 fileList,
                 fileFormat={separator:"\t", skiplines:0, ignoreRowCol:(0,0)},
                 title=None,
                 showColAxType=["lin","lin","lin","lin"],
                 showColAxLim=[None,None,None,None],
                 showColLabel= ["","X","Y", "Z",],
                 showColLabelUnit=["","X","Y", "Z",],
                 averageMedian=False,
                 errors=False,
                )
        
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
        if self.filenamePrefix is not None:
            string=self.filenamePrefix+self.fill+string
        return string+option
        
        
    def __importData(self):
        if not self.dataImported:
            self.dataList=[[Data(fileToNpArray(measurement, **self.fileFormat)[0], xCol=self.xCol, yCol=self.showCol) for measurement in sample] for sample in self.fileList]
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
        try:
            qtyCol=len(dataList[0][0].getData()[0])
            dataColList=[[[data.getData()[:,m] for data in deviceData] for deviceData in dataList] for m in range(0,qtyCol)]
            avgColList=[[np.average(element, axis=0) for element in column] for column in dataColList]
            avgList=[[avgColList[m][n] for m in range(0,qtyCol)] for n in range(0,len(dataList))]
            avgDataList=[Data.mergeData(avgData) for avgData in avgList]
            devDataList=[np.sqrt(np.sum([np.square(np.subtract(data,tempData.getData())) for tempData in deviceData],axis=0)/len(deviceData)) for data, deviceData in zip(avgDataList, dataList)]
            avgData=[Data(data, xCol=self.xCol, yCol=self.showCol) for data in avgDataList]
            devData=[Data(data, xCol=self.xCol, yCol=self.showCol) for data in devDataList]
        except (ValueError,IndexError):
            warnings.warn("Unsupported Input for Error estimation given")
            avgData, devData = self.concentenate_data()
        return avgData, devData

    #
    #
    #   returns List with median averaged values, List with standarddeviation values for each sample
    @functools.lru_cache()
    def processMedian(self, dataList=None):
        if dataList is None:
            dataList=self.dataList
        try:
            qtyCol=len(dataList[0][0].getData()[0])
            dataColList=[[[data.getData()[:,m] for data in deviceData] for deviceData in dataList] for m in range(0,qtyCol)]
            medColList=[[np.median(element, axis=0) for element in column] for column in dataColList]
            medList=[[medColList[m][n] for m in range(0,qtyCol)] for n in range(0,len(dataList))]
            medDataList=[Data.mergeData(medData) for medData in medList]
            devDataList=[np.sqrt(np.sum([np.square(np.subtract(data,tempData.getData())) for tempData in deviceData],axis=0)/len(deviceData)) for data, deviceData in zip(medDataList, dataList)]
            medData=[Data(data, xCol=self.xCol, yCol=self.showCol) for data in medDataList]
            devData=[Data(data, xCol=self.xCol, yCol=self.showCol) for data in devDataList]
        except (ValueError,IndexError):
            warnings.warn("Unsupported Input for Error estimation given")
            medData, devData = self.concentenate_data()
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
            logErrMin=[np.nan_to_num(np.minimum(err, mind)) for err,mind in zip(symErr,minErr)]
            logErrMax=[np.nan_to_num(np.minimum(err, maxi)) for err,maxi in zip(symErr,maxErr)]
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
            device_data=Data(np.vstack(dataArray), xCol=self.xCol, yCol=self.showCol)
            device_data.removeDoubles()
            new_dataList.append(device_data)
            new_devia_dataList.append(Data(np.zeros(device_data.getData().shape), xCol=self.xCol, yCol=self.showCol))
        return new_dataList, new_devia_dataList
    
    def processAverage(self):
        if not self.averageProcessed:
            if not self.concentenate_files_instead_of_avg:
                if self.averageMedian:
                    expectData, deviaData = self.processMedian()
                else:
                    expectData, deviaData = self.processAvg()
                self.expectData=expectData
                self.deviaData=deviaData
                self.logErr=self.calcLogErr(self.expectData,self.deviaData)
            else:
                self.expectData, self.deviaData=self.concentenate_data()
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
    
    def processPlot(self):
        expectData=self.processAverage()[0]
        colors=self.colors
        labels=self.labels
        xCol=self.xCol
        showCol=self.showCol
        showCol2=self.showCol2
        colorOffset=self.colorOffset
        ax=self.ax
        ax2=self.ax2
        
        if self.iterLinestyles:
            linestyles=self.linestyles
            linestyleOffset=self.linestyleOffset
            ax1color=[self.ax1color]*len(expectData)
            ax2color=[self.ax2color]*len(expectData)
        else:
            linestyles=[self.ls]*len(expectData)
            linestyleOffset=0
            ax1color=self.colors
            ax2color=self.colors
        for n in range(0,len(expectData)):
            try:
                labelY=labels[n]+" "+self.showColLabel[self.showCol]
            except TypeError:
                labelY=[labels[n]+" "+self.showColLabel[singleShowCol] for singleShowCol in self.showCol]
            if self.show[n][0]:
                if self.errors[n][0]:
                    try:
                        AX=ax.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=showCol), yerr=[self.logErr[self.errorTypeDown][n][:,showCol-1],self.logErr[self.errorTypeUp][n][:,showCol-1]], c=ax1color[n], capsize=self.capsize, capthick=self.capthick , ls=linestyles[n], label=labelY, errorevery=self.showErrorOnlyEvery)
                    except TypeError:
                        for singleShowCol,singleLabelY in zip(showCol,labelY):
                            AX=ax.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=singleShowCol), yerr=[self.logErr[self.errorTypeDown][n][:,showCol-1],self.logErr[self.errorTypeUp][n][:,showCol-1]], c=ax1color[n], capsize=self.capsize, capthick=self.capthick , ls=linestyles[n], label=singleLabelY, errorevery=self.showErrorOnlyEvery)
                else:
                    try:
                        AX=ax.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=showCol), c=ax1color[n], ls=linestyles[n], label=labelY)
                    except TypeError:
                        for singleShowCol,singleLabelY in zip(showCol,labelY):
                            AX=ax.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=singleShowCol), c=ax1color[n], ls=linestyles[n], label=singleLabelY)
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
            if self.show[n][1] and self.showCol2 is not 0:
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
                        AX2=ax2.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=showCol2), yerr=[self.logErr[self.ax2errorTypeDown][n][:,showCol2-1],self.logErr[self.ax2errorTypeUp][n][:,showCol2-1]], capsize=self.capsize, capthick=self.capthick, c=ax2color[n], ls=linestyles[n+linestyleOffset], label=labelZ,  errorevery=self.showErrorOnlyEvery)
                    except TypeError:
                        for singleShowCol2,singleLabelZ in zip(showCol2,labelZ):
                            AX2=ax2.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=singleShowCol2), yerr=[self.logErr[self.ax2errorTypeDown][n][:,showCol2-1],self.logErr[self.ax2errorTypeUp][n][:,showCol2-1]], capsize=self.capsize, capthick=self.capthick, c=ax2color[n], ls=linestyles[n+linestyleOffset], label=singleLabelZ,  errorevery=self.showErrorOnlyEvery)
                else:
                    try:
                        AX2=ax2.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=showCol2), c=ax2color[n], ls=linestyles[n+linestyleOffset], label=labelZ)
                    except TypeError:
                        for singleShowCol2,singleLabelZ in zip(showCol2,labelZ):
                            AX2=ax2.errorbar(*expectData[n].getSplitData2D(xCol=xCol, yCol=singleShowCol2), c=ax2color[n], ls=linestyles[n+linestyleOffset], label=singleLabelZ)
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
    
    def processAllAndExport(self, **kwargs):
        self.exportAllData(**kwargs)
    
    def exportAllData(self, fileEnd=".csv", colSep=",", fill=None, errorTypes=None, errorString="Error of ", expectData=None, errData=None, noError=False):
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
                           
        for l in range(0,len(self.fileList)):
            file = open(self.name.replace(" ","")+fill+self.labels[l].replace(" ","")+fileEnd,"w")
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
            
    
    def doPlotName(self):
        return self.doPlot()[1]
    
class ListShapeException(Exception):
    pass 

class FitException(Exception):
    def __init__(self,index, props, err, params):
        self.message="Error @ {} with Properties: {} \n Message: {} with parameters {}".format(index,props,str(err),str(params)) 
        