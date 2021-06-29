
# coding: utf-8

# In[1]:


import matplotlib as mpl
#mpl.use("pgf")
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
                 showColLabel= ["","Time","Normalised Intensity"],
                 showColLabelUnit=["",
                  "Time ({}s)",
                  "Normalised Intensity",             
                 ],
                 averageMedian=False,
                 errors=False,
                 fitColors=['#000000','#1f77b4','#d62728','#2ca02c','#9467bd','#8c564b','#e377c2','#7f7f7f','#ff7f0e','#bcbd22','#17becf','#f8e520'],
                 bgfile=None,
                 normalize_peak=True,
                 set_peak_to_zero=True,
                 time_domain="n",
                 fse="Decay with \n A = {:3.0f}\\,\\% \\& $\\tau$ ~= {:3.0f}\\,{}s",
                 **kwargs
                ):
        Plot.__init__(self, name, fileList, averageMedian=averageMedian, showColAxType=showColAxType, showColAxLim=showColAxLim, showColLabel=showColLabel, showColLabelUnit=showColLabelUnit, fileFormat=fileFormat, errors=errors, fitColors=fitColors, partialFitLabels=["Partial mono-exponential fit"], **kwargs)
        #dyn inits
        if title is None:
            self.title=name
        else:
            self.title=title
        self.validYCol=validYCol
        self.bgfile=bgfile
        self.normalize_peak=normalize_peak
        self.set_peak_to_zero=set_peak_to_zero
        self.time_domain=time_domain
        self.showColLabelUnit[1]=showColLabelUnit[1].format(time_domain)
        self.fse=fse
        #self.dataList=self.importData()

    def processFileName(self, option=".pdf"):
        if self.filename is None:
            string=self.name.replace(" ","")+self.fill+"lifetime"
        else:
            string=self.filename
        if not self.scaleX == 1:
            string+=self.fill+"scaledWith{:03.0f}Pct".format(self.scaleX*100)
        if self.filenamePrefix is not None:
            string=self.filenamePrefix+string
        if not self.normalize_peak:
            string+=self.fill+"not"
            string+=self.fill+"normalised"
        return string+option
    
    def processData(self):
        yCol_l= self.validYCol
        if not self.dataProcessed:
            if self.bgfile is not None:
                bg=fileToNpArray(self.bgfile, **self.fileFormat)[0][:,1]
            else:
                bg = 0
            for device in self.dataList:
                for data in device:
                    if not isinstance(yCol_l, list):
                        yCol_l=[yCol_l]
                    if not isinstance(self.normalize_peak, list):
                        self.normalize_peak=[self.normalize_peak]
                    if not isinstance(self.set_peak_to_zero, list):
                        self.set_peak_to_zero=[self.set_peak_to_zero]
                    done=False
                    for yCol,normalize_peak,set_peak_to_zero in zip(yCol_l,self.normalize_peak,self.set_peak_to_zero):
                        time,intens= data.getSplitData2D(xCol=1, yCol=yCol)[0], data.getSplitData2D(xCol=1,yCol=yCol)[1]-bg
                        #data.setData(Data.mergeData((time, intens, intens)))
                        data.processData(self.noNegatives, yCol=yCol)
                        if normalize_peak:
                            data.processData(self.normalize, yCol=yCol)
                        if set_peak_to_zero and not done:
                            max_value=np.amax(intens)    
                            indices = np.where(intens == max_value)
                            data.processData(self.shift, x=True, y=False, index=indices[0][0])
                            done=True
                    data.limitData(xLim=self.xLimOrig)        
                self.dataProcessed=True 
            return self.dataList

    def plotExp(self,fitter,n):
        xdata=fitter.CurveData.getSplitData2D()[0]
        ydata1=self.exp(xdata, *fitter.params[0:2], 0)
        ydata3=xdata/xdata*fitter.params[2]
        textPos=fitter.textPos
        amp=-fitter.params[1]
        tp1=textPos
        self.ax.errorbar(xdata, ydata3, c=self.fitColors[n+1], ls=self.fitLs, label="Offset", alpha=self.fitAlpha)
        self.ax.errorbar(xdata, ydata1, c=self.fitColors[n+2], ls=self.fitLs, label="Partial mono-exponential fit", alpha=self.fitAlpha)
        self.handleDesc(fitter, n=n+2, xsy=(0,np.amax(ydata1)), tp=tp1)  
        
        
    def plotDoubleExp(self,fitter,n):
        xdata=fitter.CurveData.getSplitData2D()[0]
        ydata1=self.exp(xdata, *fitter.params[0:2], 0)
        ydata2=self.exp(xdata, *fitter.params[3:5], 0)
        ydata3=xdata/xdata*fitter.params[2]
        textPos=fitter.textPos
        textPos2=[fitter.textPos[0]+1*np.amax(xdata)/10,fitter.textPos[1]*0.3]
        amp=-fitter.params[1]
        amp2=-fitter.params[4]
        if amp > amp2:
            tp1=textPos2
            tp2=textPos
        else:
            tp1=textPos
            tp2=textPos2
        self.ax.errorbar(xdata, ydata3, c=self.fitColors[n+1], ls=self.fitLs, label="Offset", alpha=self.fitAlpha)
        self.ax.errorbar(xdata, ydata1, c=self.fitColors[n+2], ls=self.fitLs, label="Partial mono-exponential fit", alpha=self.fitAlpha)
        self.ax.errorbar(xdata, ydata2, c=self.fitColors[n+3], ls=self.fitLs, label="Partial mono-exponential fit", alpha=self.fitAlpha)
        self.handleDesc(fitter, n=n+2, xsy=(0,np.amax(ydata1)), tp=tp1)
        self.handleDesc(fitter, n=n+3, param_pos=3, xsy=(0,np.amax(ydata2)), tp=tp2)
        #fse=self.fse
        #se=fse.format(np.round(fitter.params[1]*100,decimals=0),np.round(fitter.params[0],decimals=0),self.time_domain)
        #self.ax.annotate(s=se, size=self.customFontsize[2], xy=(0,np.amax(ydata1)), xytext=tp1, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n+2], edgecolor=self.fitColors[n+2], linewidth=mpl.rcParams["lines.linewidth"]))
        #se2=fse.format(np.round(fitter.params[4]*100,decimals=0),np.round(fitter.params[3],decimals=0),self.time_domain)
        #self.ax.annotate(s=se2, size=self.customFontsize[2], xy=(0,np.amax(ydata2)), xytext=tp2, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n+3], edgecolor=self.fitColors[n+3], linewidth=mpl.rcParams["lines.linewidth"]))
        
    def plotTripleExp(self,fitter,n):
        xdata=fitter.CurveData.getSplitData2D()[0]
        ydata1=self.exp(xdata, *fitter.params[0:2], 0)
        ydata2=self.exp(xdata, *fitter.params[3:5], 0)
        ydata3=self.exp(xdata, *fitter.params[5:7], 0)
        ydata4=xdata/xdata*fitter.params[2]
        textPos=fitter.textPos
        textPos2=[fitter.textPos[0]+1*np.amax(xdata)/10,fitter.textPos[1]*0.3]
        textPos3=[fitter.textPos[0]+2*np.amax(xdata)/10,fitter.textPos[1]*0.08]
        amp=-fitter.params[1]
        amp2=-fitter.params[4]
        amp3=-fitter.params[6]
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
        
        self.ax.errorbar(xdata, ydata4, c=self.fitColors[n+1], ls=self.fitLs, label="Offset", alpha=self.fitAlpha)
        self.ax.errorbar(xdata, ydata1, c=self.fitColors[n+2], ls=self.fitLs, label="Partial mono-exponential fit", alpha=self.fitAlpha)
        self.ax.errorbar(xdata, ydata2, c=self.fitColors[n+3], ls=self.fitLs, label="Partial mono-exponential fit", alpha=self.fitAlpha)
        self.ax.errorbar(xdata, ydata3, c=self.fitColors[n+4], ls=self.fitLs, label="Partial mono-exponential fit", alpha=self.fitAlpha)
        self.handleDesc(fitter, n=n+2, xsy=(0,np.amax(ydata1)), tp=tp1)
        self.handleDesc(fitter, n=n+3, param_pos=3, xsy=(0,np.amax(ydata2)), tp=tp2)
        self.handleDesc(fitter, n=n+4, param_pos=5, xsy=(0,np.amax(ydata3)), tp=tp3)
        #fse=self.fse
        #se=fse.format(np.round(fitter.params[1]*100,decimals=0),np.round(fitter.params[0],decimals=0),self.time_domain)
        #self.ax.annotate(s=se, size=self.customFontsize[2], xy=(0,np.amax(ydata1)), xytext=tp1, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n+2], edgecolor=self.fitColors[n+2], linewidth=mpl.rcParams["lines.linewidth"]))
        #se2=fse.format(np.round(fitter.params[4]*100,decimals=0),np.round(fitter.params[3],decimals=0),self.time_domain)
        #self.ax.annotate(s=se2, size=self.customFontsize[2], xy=(0,np.amax(ydata2)), xytext=tp2, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n+3], edgecolor=self.fitColors[n+3], linewidth=mpl.rcParams["lines.linewidth"]))
        #se3=se2=fse.format(np.round(fitter.params[6]*100,decimals=0),np.round(fitter.params[5],decimals=0),self.time_domain)
        #self.ax.annotate(s=se3, size=self.customFontsize[2], xy=(0,np.amax(ydata3)), xytext=tp3, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n+4], edgecolor=self.fitColors[n+4], linewidth=mpl.rcParams["lines.linewidth"]))
    
        
    def importData(self):
        self.dataList=[[Data(fileToNpArray(pixel, **self.fileFormat)[0]) for pixel in device] for device in self.fileList]
        return self.dataList
    
    
    def handleDesc(self, fitter, n=2, param_pos=0, xsy=None, tp=None):
        ax=self.ax
        if fitter.params[0] < 1 or self.time_domain=="p":
            timedomain = "p"
            lifetime=fitter.params[param_pos]*1000
        else:
            lifetime=fitter.params[param_pos]
            timedomain=self.time_domain
        #if self.xCol!=4:
            #se=fitter.desc.format(np.round(fitter.params[self.xParamPos]))
       # else:

        try:
            se=fitter.desc.format(np.round(fitter.params[param_pos+1]*100,decimals=0),np.round(lifetime,decimals=0),timedomain)
        except:
            se=self.fse.format(np.round(fitter.params[param_pos+1]*100,decimals=0),np.round(lifetime,decimals=0),timedomain)

        sze=self.customFontsize[2]
        if xsy is None:
            xsy=(fitter.params[self.xParamPos],np.amax(fitter.CurveData.getSplitData2D()[1])-0.1*np.amax(fitter.CurveData.getSplitData2D()[1]))
        if tp is None:
            tp=fitter.textPos
        arprps=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n], edgecolor=self.fitColors[n], linewidth=mpl.rcParams["lines.linewidth"])
        ax.annotate(s=se, size=sze, xy=xsy, xytext=tp, arrowprops=arprps)
    
    
    def afterPlot(self):
        
        for n in range(0,len(self.expectData)):
            try:
                if self.fitterList[n] is not None:
                    if type(self.fitterList[n]) is list:
                        for fitter in self.fitterList[n]:
                            if fitter.function == self.exp:
                                self.plotExp(fitter,n)
                            if fitter.function == self.doubleExp:
                                self.plotDoubleExp(fitter,n)
                            if fitter.function == self.tripleExp:
                                self.plotTripleExp(fitter,n)
                    else:
                        if self.fitterList[n].function == self.exp:
                            self.plotExp(self.fitterList[n],n)
                        if self.fitterList[n].function == self.doubleExp:
                            self.plotDoubleExp(self.fitterList[n],n)
                        if self.fitterList[n].function == self.tripleExp:
                            self.plotTripleExp(self.fitterList[n],n)

            except Exception as e:
                print(e)
                


    
    
   