
# coding: utf-8

# In[1]:


import matplotlib as mpl
#mpl.use("pgf")
import matplotlib.pyplot
import numpy as np
import scipy.interpolate as inter
import pandas as pd
import copy
import sys
import os
import string
import warnings
import functools
import copy
import inspect
import datetime
from matplotlib import rc
from Filereader import fileToNpArray, comma_str_to_float
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
    
    default_streak_file_format=dict(index_col=0, header=0 ,sep="\t", encoding="utf-8", converters={0:comma_str_to_float})
    default_phelos_file_format=dict(index_col=0, header=None ,sep="\t", comment='#', encoding="iso-8859-1")
    
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
    
    @classmethod
    def parse_streak_data(cls, filenames, locs, fileformat=default_streak_file_format, spectrometer_thresh=1):
        data=[]
        for filename,loc in zip(filenames,locs):
            spectra_d=pd.read_csv(filename[0], **fileformat)
            spectra_d.columns=[comma_str_to_float(col) for col in spectra_d.columns]
            i=0
            for col in spectra_d.columns:
                if col <= loc:
                    break
                i+=1
            pre_array=spectra_d.iloc[:,i]
            a_np=pre_array.to_numpy()
            a_min=a_np.min()
            a_np=a_np-a_min
            a_np[a_np == 0.0]=spectrometer_thresh
            a_max=a_np.max()
            a_np=a_np/a_max
            t=pre_array.index.to_numpy()
            data.append((t,a_np))
        return data
    
    @classmethod
    def parse_phelos_datetime(cls, phelos_timestring):
        month, day, year = phelos_timestring[-1].split("/")
        hour, minute, second = phelos_timestring[0].split(":")
        year=int(year)
        month= int(month)
        day = int(day)
        if phelos_timestring[1]=="PM":
            hour= int(hour)+12
        else:
            hour= int(hour)
        minute= int(minute)
        microsecond= int(second.split(".")[1])*1000
        second= int(second.split(".")[0])
        return datetime.datetime(year, month, day, hour, minute, second, microsecond)
    
    @classmethod    
    def parse_phelos_data(cls, filenames, locs, fileformat=default_phelos_file_format, spectrometer_thresh=5*10**-8):
        data=[]
        for filename,loc in zip(filename,locs):

            with open(filename[0], encoding=fileformat["encoding"]) as file:
                file2=file.readlines()
                desc=[string for string in file2 if string.startswith("#")]
            i=0
            for line in desc:
                if line.startswith('# Measured between'):
                    meas_start=desc[i+1]
                    meas_start_time=self.parse_phelos_datetime(meas_start.split(" ")[1:4])
                    meas_end=desc[i+2]
                    meas_end_time=self.parse_phelos_datetime(meas_end.split(" ")[1:4])
                elif line.startswith('# Current'):
                    current=line.split(" ")[2:]
                elif line.startswith('# Sweep'):
                    steps=int(line.split("|")[1].split(" ")[3])
                i+=1
            timestep=(meas_end_time-meas_start_time)/steps
            timesteps=[meas_start_time+timestep*step for step in range(1,steps+1)]

            spectra_d=pd.read_csv(filename[0], **fileformat)
            spectra_d=spectra_d.drop(columns=np.arange(2,len(spectra_d.columns-1),2))
            spectra_d.columns=timesteps
            i=0
            for row in spectra_d.index:
                if row >= loc:
                    break
                i+=1
            a=spectra_d.iloc[i]
            a_np=a.to_numpy()
            a_np=a_np[a_np > spectrometer_thresh]
            a_max=a_np.max()
            a_np=np.abs(a_np)
            a_np=a_np/a_max
            timesteps=timesteps[:len(a_np)]
            t=[(timestep-timesteps[0] ).total_seconds() for timestep in timesteps]
            data.append((t,a_np))
        return data
    
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
                 fse="Decay with \n A = {:3.0f}\\,\\% \\& t = {:3.0f}\\,{}s",
                 parse_data_style="streak",
                 locs=None,
                 **kwargs
                ):
        self.parse_data_style=parse_data_style
        self.locs=locs
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
            self.processFileName_makedirs()
            if self.filenamePrefix[-1] == os.sep:
                string=self.filenamePrefix+string
            else:
                string=self.filenamePrefix+self.fill+string
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
        if not self.dataImported:
            dataList=[]
            filenames=self.fileList
            if self.parse_data_style == "streak":
                if self.locs is None:
                    raise Exception("Locs unspecified")
                locs=self.locs
                if self.fileFormat is None:
                    data=self.parse_streak_data(filenames, locs)
                else:
                    data=self.parse_streak_data(filenames, locs, fileformat=self.fileFormat)
            elif self.parse_data_style == "phelos":
                if self.locs is None:
                    raise Exception("Locs unspecified")
                locs=self.locs
                if self.fileformat is None:
                    data=self.parse_phelos_data(filenames, locs)
                else:
                    data=self.parse_phelos_data(filenames, locs, fileformat=self.fileFormat)
            else:
                raise Exception("No such parse style")
            for data_tup in data:
                dataList.append([Data(Data.mergeData2D(*data_tup))])
            self.dataList=dataList
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
        ax.annotate(text=se, size=sze, xy=xsy, xytext=tp, arrowprops=arprps)
    
    
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
                


    
    
   