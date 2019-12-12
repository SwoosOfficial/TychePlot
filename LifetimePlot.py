
# coding: utf-8

# In[1]:


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


# In[2]:


class LifetimePlot(Plot):
    #NaturKonst
    e=1.6*10**-19 #C
    c=2.99*10**17 #nm/s
    h=6.63*10**-34 #J*s
    #ProgKonst
    chars=list(string.ascii_uppercase) #alphabetUppercase
    convFac=(h*c)/e #eV*nm
    
    @classmethod
    def noNegatives(cls,a):
        return np.maximum(a,np.zeros(len(a), dtype=np.float64))
    
    @classmethod
    def normalize(cls,a):
        b=a-np.amin(a)
        return b/np.amax(b, axis=0)

    @classmethod
    def shift(cls, data, index):
        return data-data[index]
    
    @classmethod
    def exp(cls, x, tau, amp, offset):
        return amp*np.exp(-x/tau)+offset
    
    @classmethod
    def doubleExp(cls, x, tau, amp, offset, tau2, amp2):
        return amp*np.exp(-x/tau)+amp2*np.exp(-x/tau2)+offset
    
    @classmethod
    def tripleExp(cls, x, tau, amp, offset, tau2, amp2, tau3, amp3):
        return amp*np.exp(-x/tau)+amp2*np.exp(-x/tau2)+amp3*np.exp(-x/tau3)+offset
    
    @classmethod
    def twoExpSplines(cls, x, end1, start2, tau, amp, offset, tau2, amp2, offset2):
        return amp*np.exp(-x/tau)+offset*np.heaviside(end1-x,0)+amp2*np.exp(-x/tau2)+offset2*np.heaviside(x-start2,0)
    
    def __init__(self,
                 name,
                 fileList,
                 fileFormat={"separator":"\t", "skiplines":10},
                 title=None,
                 validYCol=2,
                 showColAxType=["lin","lin","log"],
                 showColAxLim=[None,None,None],
                 showColLabel= ["","Time","Normalized Intensity"],
                 showColLabelUnit=["",
                  "Time ({}s)",
                  "Normalized Intensity",             
                 ],
                 averageMedian=False,
                 errors=False,
                 xParamPos=0,
                 rainbowMode=False,
                 fitColors=['#1f77b4','#d62728','#2ca02c','#9467bd','#8c564b','#e377c2','#7f7f7f','#ff7f0e','#bcbd22','#17becf','#f8e520'],
                 bgfile=None,
                 normalize_peak=True,
                 set_peak_to_zero=True,
                 time_domain="n",
                 **kwargs
                ):
        Plot.__init__(self, name, fileList, averageMedian=averageMedian, showColAxType=showColAxType, showColAxLim=showColAxLim, showColLabel=showColLabel, showColLabelUnit=showColLabelUnit, fileFormat=fileFormat, errors=errors, fitColors=fitColors, partialFitLabels=["Partial mono-Gaussian fit"], **kwargs)
        #dyn inits
        if title is None:
            self.title=name
        else:
            self.title=title
        self.validYCol=validYCol
        self.xParamPos=xParamPos
        self.rainbowMode=rainbowMode
        self.bgfile=bgfile
        self.normalize_peak=normalize_peak
        self.set_peak_to_zero=set_peak_to_zero
        self.showColLabelUnit[1]=showColLabelUnit[1].format(time_domain)
        #self.dataList=self.importData()
   
    def processFileName(self, option=".pdf"):
        if self.filename is None:
            string=self.name.replace(" ","")+self.fill+"lifetime"
        else:
            string=self.filename
        if not self.scaleX is 1:
            string+=self.fill+"scaledWith{:03.0f}Pct".format(self.scaleX*100)
        if self.filenamePrefix is not None:
            string=self.filenamePrefix+string
        if not self.normalize_peak:
            string+=self.fill+"not"
            string+=self.fill+"normalised"
        return string+option
    
    def processData(self):
        yCol= self.validYCol
        if not self.dataProcessed:
            if self.bgfile is not None:
                bg=fileToNpArray(self.bgfile, **self.fileFormat)[0][:,1]
            else:
                bg = 0
            for device in self.dataList:
                for data in device:
                    time,intens= data.getSplitData2D(xCol=1, yCol=yCol)[0], data.getSplitData2D(xCol=1,yCol=yCol)[1]-bg
                    #data.setData(Data.mergeData((time, intens, intens)))
                    data.processData(self.noNegatives, yCol=2)
                    if self.normalize:
                        data.processData(self.normalize, yCol=2)
                    if self.set_peak_to_zero:
                        max_value=np.amax(intens)    
                        indices = np.where(intens == max_value)
                        data.processData(self.shift, x=True, y=False, index=indices[0][0])
                    data.limitData(xLim=self.xLimOrig)        
                self.dataProcessed=True 
            return self.dataList
  
    
    def plotDoubleGauss(self,fitter,n):
        xdata=fitter.CurveData.getSplitData2D()[0]
        ydata1=self.exp(xdata, *fitter.params[0:3])
        ydata2=self.exp(xdata, *fitter.params[3:5], fitter.params[2])
        textPos=fitter.textPos
        textPos2=[fitter.textPos[0],fitter.textPos[1]+0.15]
        amp=1#fitter.params[1]/(np.sqrt(2*np.pi*fitter.params[2]**2))
        amp2=2#fitter.params[1+3]/(np.sqrt(2*np.pi*fitter.params[2+3]**2))
        if amp > amp2:
            tp1=textPos2
            tp2=textPos
        else:
            tp1=textPos
            tp2=textPos2
        self.ax.errorbar(xdata, ydata1, c=self.fitColors[n+1], ls=self.fitLs, label="Partial mono-Gaussian fit", alpha=self.fitAlpha)
        se="Emission at \n{:3.0f}\\,nm / {:3.2f}\\,eV".format(np.round(self.convFac/fitter.params[self.xParamPos],decimals=0),np.round(fitter.params[self.xParamPos],decimals=2))
        self.ax.annotate(s=se, size=self.customFontsize[2], xy=(fitter.params[self.xParamPos],np.amax(ydata1)-0.1*np.amax(ydata1)), xytext=tp1, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n+1], edgecolor=self.fitColors[n+1], linewidth=mpl.rcParams["lines.linewidth"]))
        self.ax.errorbar(xdata, ydata2, c=self.fitColors[n+2], ls=self.fitLs, label="Partial mono-Gaussian fit", alpha=self.fitAlpha)
        se2="Emission at \n{:3.0f}\\,nm / {:3.2f}\\,eV".format(np.round(self.convFac/fitter.params[self.xParamPos+3],decimals=0),np.round(fitter.params[self.xParamPos+3],decimals=2))
        self.ax.annotate(s=se2, size=self.customFontsize[2], xy=(fitter.params[self.xParamPos+3],np.amax(ydata2)-0.1*np.amax(ydata2)), xytext=tp2, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n+2], edgecolor=self.fitColors[n+2], linewidth=mpl.rcParams["lines.linewidth"]))
        
    def plotTripleGauss(self,fitter,n):
        xdata=fitter.CurveData.getSplitData2D()[0]
        ydata1=self.exp(xdata, *fitter.params[0:3])
        ydata2=self.exp(xdata, *fitter.params[3:5], fitter.params[2])
        ydata3=self.exp(xdata, *fitter.params[5:7], fitter.params[2])
        textPos=fitter.textPos
        textPos2=[fitter.textPos[0],fitter.textPos[1]+0.15]
        textPos3=[fitter.textPos[0],fitter.textPos[1]+0.3]
        amp=1#fitter.params[1]/(np.sqrt(2*np.pi*fitter.params[2]**2))
        amp2=2#fitter.params[1+3]/(np.sqrt(2*np.pi*fitter.params[2+3]**2))
        amp3=3#fitter.params[1+6]/(np.sqrt(2*np.pi*fitter.params[2+6]**2))
        if amp > amp2:
            if amp > amp3:
                tp1=textPos3
                if amp2 > amp3:
                    tp2=textPos2
                    tp3=textPos
                else:
                    tp2=textPos
                    tp3=textPos2
            else:
                tp3=textPos3
                tp1=textPos2
                tp2=textPos
                
        else:
            if amp2 > amp3:
                tp2=textPos3
                if amp > amp3:
                    tp1=textPos2
                    tp3=textPos
                else:
                    tp3=textPos2
                    tp1=textPos
            else:
                tp3=textPos3
                tp2=textPos2
                tp1=textPos
        #fit1
        self.ax.errorbar(xdata, ydata1, c=self.fitColors[n+1], ls=self.fitLs, label="Partial mono-Gaussian fit", alpha=self.fitAlpha)
        se="Emission at \n{:3.0f}\\,nm / {:3.2f}\\,eV".format(np.round(self.convFac/fitter.params[self.xParamPos],decimals=0),np.round(fitter.params[self.xParamPos],decimals=2))
        self.ax.annotate(s=se, size=self.customFontsize[2], xy=(fitter.params[self.xParamPos],np.amax(ydata1)-0.1*np.amax(ydata1)), xytext=tp1, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n+1], edgecolor=self.fitColors[n+1], linewidth=mpl.rcParams["lines.linewidth"]))
        #fit2
        self.ax.errorbar(xdata, ydata2, c=self.fitColors[n+2], ls=self.fitLs, label="Partial mono-Gaussian fit", alpha=self.fitAlpha)
        se2="Emission at \n{:3.0f}\\,nm / {:3.2f}\\,eV".format(np.round(self.convFac/fitter.params[self.xParamPos+3],decimals=0),np.round(fitter.params[self.xParamPos+3],decimals=2))
        self.ax.annotate(s=se2, size=self.customFontsize[2], xy=(fitter.params[self.xParamPos+3],np.amax(ydata2)-0.1*np.amax(ydata2)), xytext=tp2, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n+2], edgecolor=self.fitColors[n+2], linewidth=mpl.rcParams["lines.linewidth"]))
        #fit3
        self.ax.errorbar(xdata, ydata3, c=self.fitColors[n+3], ls=self.fitLs, label="Partial mono-Gaussian fit", alpha=self.fitAlpha)
        se3="Emission at \n{:3.0f}\\,nm / {:3.2f}\\,eV".format(np.round(self.convFac/fitter.params[self.xParamPos+6],decimals=0),np.round(fitter.params[self.xParamPos+6],decimals=2))
        self.ax.annotate(s=se3, size=self.customFontsize[2], xy=(fitter.params[self.xParamPos+6],np.amax(ydata3)-0.1*np.amax(ydata3)), xytext=tp3, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n+3], edgecolor=self.fitColors[n+3], linewidth=mpl.rcParams["lines.linewidth"]))
    
        
    def importData(self):
        self.dataList=[[Data(fileToNpArray(pixel, **self.fileFormat)[0]) for pixel in device] for device in self.fileList]
        return self.dataList
    
    
    
    def afterPlot(self):
        ax=self.ax
        for n in range(0,len(self.expectData)):
            try:
                if self.fitterList[n] is not None:
                    if type(self.fitterList[n]) is list:
                        for fitter in self.fitterList[n]:
                            if fitter.function == self.doubleGauss:
                                self.plotDoubleGauss(fitter,n)
                            if fitter.function == self.tripleGauss:
                                self.plotTripleGauss(fitter,n)
                            #annotation
                            if fitter.desc != None:
                                if self.xCol!=4:
                                    se=fitter.desc.format(np.round(fitter.params[self.xParamPos]))
                                else:
                                    try:
                                           se=fitter.desc.format(np.round(self.convFac/fitter.params[self.xParamPos],decimals=0),np.round(fitter.params[self.xParamPos],decimals=1))
                                    except IndexError:
                                        se=fitter.desc.format(np.round(self.convFac/fitter.params[self.xParamPos]))
                                ax.annotate(s=se, size=self.customFontsize[2], xy=(fitter.params[self.xParamPos],np.amax(fitter.CurveData.getSplitData2D()[1])-0.1*np.amax(fitter.CurveData.getSplitData2D()[1])), xytext=fitter.textPos, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n], edgecolor=self.fitColors[n], linewidth=mpl.rcParams["lines.linewidth"]))
                    else:
                        if self.fitterList[n].function == self.doubleGauss:
                            self.plotDoubleGauss(self.fitterList[n],n)
                        if self.fitterList[n].function == self.tripleGauss:
                            self.plotTripleGauss(self.fitterList[n],n)
                        #annotation
                        if self.fitterList[n].desc != None:
                            fitter=self.fitterList[n]
                            if self.xCol!=4:
                                se=fitter.desc.format(np.round(fitter.params[self.xParamPos]))
                            else:
                                try:
                                    se=fitter.desc.format(np.round(self.convFac/fitter.params[self.xParamPos],decimals=0),np.round(fitter.params[self.xParamPos],decimals=1))
                                except IndexError:
                                    se=fitter.desc.format(np.round(self.convFac/fitter.params[self.xParamPos],decimals=0))
                            sze=self.customFontsize[2]
                            xsy=(fitter.params[self.xParamPos],np.amax(fitter.CurveData.getSplitData2D()[1])-0.1*np.amax(fitter.CurveData.getSplitData2D()[1]))
                            arprps=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n], edgecolor=self.fitColors[n], linewidth=mpl.rcParams["lines.linewidth"])
                            ax.annotate(s=se, size=sze, xy=xsy, xytext=fitter.textPos, arrowprops=arprps)
            except Exception as e:
                pass
                


    
    
   