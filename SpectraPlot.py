
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


class SpectraPlot(Plot):
    #NaturKonst
    e=1.6*10**-19 #C
    K_m=683 #lm/W
    c=2.99*10**17 #nm/s
    h=6.63*10**-34 #J*s
    #ProgKonst
    chars=list(string.ascii_uppercase) #alphabetUppercase
    convFac=(h*c)/e #eV*nm
    
    @classmethod
    def wavelengthToEV(cls,wavelength,spectralRadiance):
        energy=wavelength**-1*cls.convFac #eV
        corFac=wavelength**2*cls.e/(cls.h*cls.c) #nm/eV
        spectralRadianceEnergy=spectralRadiance*corFac #W/(sr*m^2*eV)
        return (energy,spectralRadianceEnergy)
    
    @classmethod
    def noNegatives(cls,a):
        return np.maximum(a,np.zeros(len(a), dtype=np.float64))
    
    @classmethod
    def normalize(cls,a):
        b=a-np.amin(a)
        return b/np.amax(b, axis=0)

    @classmethod
    def gauss(cls, x, mu, amp, sigma):
        return amp/(np.sqrt(2*np.pi*sigma**2))*np.exp(-((x-mu)**2/(2*sigma**2)))
    
    @classmethod
    def twoGaussSplines(cls, x, mu, amp, sigma, sigma2):
        return amp*np.exp(-((x-mu)**2/(2*sigma**2)))*np.heaviside(x-mu,0)+amp*np.exp(-((x-mu)**2/(2*sigma2**2)))*np.heaviside(mu-x,0)
    

    
    def __init__(self,
                 name,
                 fileList,
                 fileFormat={"separator":";", "skiplines":82},
                 title=None,
                 validYCol=[2],
                 bgYCol=[None],
                 showColAxType=["lin","lin","lin","lin","lin","lin","lin"],
                 showColAxLim=[None,None,None,None,None,None,None],
                 showColLabel= ["","Wavelength","Normalized Intensity", "Spectral Radiance", "Energy", "Normalized Intensity", "Spectral Radiance"],
                 showColLabelUnit=["",
                  "Wavelength (nm)",
                  "Normalized Intensity",
                  "Spectral Radiance ($\\tfrac{\\mathrm{W}}{\\mathrm{sr}\\cdot \\mathrm{m}^2\\cdot \\mathrm{nm}}$)",                 
                  "Energy (eV)",
                  "Normalized Intensity", #energy scaled
                  "Spectral Radiance ($\\tfrac{\\mathrm{W}}{\\mathrm{sr}\\cdot \\mathrm{m}^2\\cdot \\mathrm{eV}}$)"                 
                 ],
                 averageMedian=False,
                 errors=False,
                 xParamPos=0,
                 **kwargs
                ):
        Plot.__init__(self, name, fileList, averageMedian=averageMedian, showColAxType=showColAxType, showColAxLim=showColAxLim, showColLabel=showColLabel, showColLabelUnit=showColLabelUnit, fileFormat=fileFormat, errors=errors, **kwargs)
        #dyn inits
        if title is None:
            self.title=name
        else:
            self.title=title
        self.bgYCol=bgYCol
        self.validYCol=validYCol
        self.xParamPos=xParamPos
        #self.dataList=self.importData()
   
    def processFileName(self, option=".pdf"):
        if self.filename is None:
            string=self.name.replace(" ","")+self.fill+"spectra"
        else:
            string=self.filename
        if not self.scaleX is 1:
            string+=self.fill+"scaledWith{:03.0f}Pct".format(self.scaleX*100)
        return string+option
    
    def processData(self):
        if not self.dataProcessed:
            for device in self.dataList:
                for data, yCol, bg in zip(device,self.validYCol, self.bgYCol):
                    try:
                        energy,specRad=self.wavelengthToEV(data.getSplitData2D(xCol=1, yCol=yCol)[0], data.getSplitData2D(xCol=1,yCol=yCol)[1]- data.getSplitData2D(xCol=1,yCol=bg)[1])
                        data.setData(Data.mergeData((data.getSplitData2D(xCol=1, yCol=yCol)[0], data.getSplitData2D(xCol=1,yCol=yCol)[1]- data.getSplitData2D(xCol=1,yCol=bg)[1],data.getSplitData2D(xCol=1,yCol=yCol)[1]- data.getSplitData2D(xCol=1,yCol=bg)[1],energy,specRad,specRad)))
                    except (IndexError,TypeError):
                        energy,specRad=self.wavelengthToEV(*data.getSplitData2D(xCol=1, yCol=yCol))
                        data.setData(Data.mergeData((data.getSplitData2D(xCol=1, yCol=yCol)[0],data.getSplitData2D(xCol=1,yCol=yCol)[1],data.getSplitData2D(xCol=1,yCol=yCol)[1],energy,specRad,specRad)))
                    data.processData(self.noNegatives, yCol=2)
                    data.processData(self.noNegatives, yCol=3)
                    data.processData(self.noNegatives, yCol=5)
                    data.processData(self.noNegatives, yCol=6)
                    data.processData(self.normalize, yCol=2)
                    data.processData(self.normalize, yCol=5)
                    data.limitData(xLim=self.xLimOrig)
            self.dataProcessed=True
        return self.dataList
    
    
    def rect(self,x,y,w,h,c):
        polygon = matplotlib.pyplot.Rectangle((x,y),w,h,color=c)
        self.ax.add_patch(polygon)
    
    
    def rainbow_fill(self,X,Y, cmap=matplotlib.pyplot.get_cmap("nipy_spectral")):
        dx = X[1]-X[0]
        S  = 380 
        N  = 675
        h= 0.01
        for n, (x,y) in enumerate(zip(X,Y)):
            if (x>N):
                color= cmap(0.9999)
            elif (x<S):
                color= cmap(0)
            else:
                color = cmap((x-S)/(N-S))
            self.rect(x,-0.035,dx,h,color)
    
    def xColTicksToXCol2Ticks(self, ticks):
        if self.xCol==1 and self.xCol2==4:
            return ["{:2.1f}".format(tick) for tick in ticks**-1*self.convFac]
        elif self.xCol==4 and self.xCol2==1:
            return ["{:3.0f}".format(tick) for tick in ticks**-1*self.convFac]
        else:
            return ticks
        
    def importData(self):
        self.dataList=[[Data(fileToNpArray(pixel, **self.fileFormat)[0]) for pixel in device] for device in self.fileList]
        return self.dataList
    
    def afterPlot(self):
        ax=self.ax
        if self.xCol==1:
            self.rainbow_fill(*self.expectData[0].getSplitData2D())
        try:
            for n in range(0,len(self.expectData)):
                if self.fitterList[n] is not None:
                    if type(self.fitterList[n]) is list:
                        for fitter in self.fitterList[n]:
                            if fitter.desc != None:
                                ax.annotate(s=fitter.desc.format(fitter.params[self.xParamPos]), size=self.customFontsize[2], xy=(fitter.params[self.xParamPos],np.amax(fitter.CurveData.getSplitData2D()[1])-0.1), xytext=fitter.textPos, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n], edgecolor=self.fitColors[n]))
                    elif self.fitterList[n].desc != None:
                        fitter=self.fitterList[n]
                        ax.annotate(s=fitter.desc.format(fitter.params[self.xParamPos]), size=self.customFontsize[2], xy=(fitter.params[self.xParamPos],np.amax(fitter.CurveData.getSplitData2D()[1])-0.1), xytext=fitter.textPos, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n], edgecolor=self.fitColors[n]))
        except Exception:
            pass
                


    
    
   