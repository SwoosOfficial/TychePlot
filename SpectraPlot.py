
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
    def doubleGauss(cls, x, mu, amp, sigma, mu2, amp2, sigma2):
        return amp/(np.sqrt(2*np.pi*sigma**2))*np.exp(-((x-mu)**2/(2*sigma**2)))+amp2/(np.sqrt(2*np.pi*sigma2**2))*np.exp(-((x-mu2)**2/(2*sigma2**2)))
    
    @classmethod
    def tripleGauss(cls, x, mu, amp, sigma, mu2, amp2, sigma2, mu3, amp3, sigma3):
        return amp/(np.sqrt(2*np.pi*sigma**2))*np.exp(-((x-mu)**2/(2*sigma**2)))+amp2/(np.sqrt(2*np.pi*sigma2**2))*np.exp(-((x-mu2)**2/(2*sigma2**2)))+amp3/(np.sqrt(2*np.pi*sigma3**2))*np.exp(-((x-mu3)**2/(2*sigma3**2)))
    
    @classmethod
    def twoGaussSplines(cls, x, mu, amp, sigma, sigma2):
        return amp*np.exp(-((x-mu)**2/(2*sigma**2)))*np.heaviside(x-mu,0)+amp*np.exp(-((x-mu)**2/(2*sigma2**2)))*np.heaviside(mu-x,0)
    
    @classmethod
    def FWHMbySigma(cls, sigma):
        return sigma*2.3548
    
    def __init__(self,
                 name,
                 fileList,
                 fileFormat={"separator":";", "skiplines":75},
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
                 rainbowMode=False,
                 fitColors=['#1f77b4','#d62728','#2ca02c','#9467bd','#8c564b','#e377c2','#7f7f7f','#ff7f0e','#bcbd22','#17becf','#f8e520'],
                 bgfile=None,
                 **kwargs
                ):
        if rainbowMode:
            kwargs.update({"legendBool":False})
        Plot.__init__(self, name, fileList, averageMedian=averageMedian, showColAxType=showColAxType, showColAxLim=showColAxLim, showColLabel=showColLabel, showColLabelUnit=showColLabelUnit, fileFormat=fileFormat, errors=errors, fitColors=fitColors, **kwargs)
        #dyn inits
        if title is None:
            self.title=name
        else:
            self.title=title
        self.bgYCol=bgYCol
        self.validYCol=validYCol
        self.xParamPos=xParamPos
        self.rainbowMode=rainbowMode
        self.bgfile=bgfile
        #self.dataList=self.importData()
   
    def processFileName(self, option=".pdf"):
        if self.filename is None:
            string=self.name.replace(" ","")+self.fill+"spectra"
        else:
            string=self.filename
        if not self.scaleX is 1:
            string+=self.fill+"scaledWith{:03.0f}Pct".format(self.scaleX*100)
        if self.filenamePrefix is not None:
            string=self.filenamePrefix+string
        return string+option
    
    def processData(self):
        if not self.dataProcessed:
            if self.bgfile is None:
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
            else:
                bg=fileToNpArray(self.bgfile, **self.fileFormat)[0][:,1]
                for device in self.dataList:
                    for data, yCol in zip(device,self.validYCol):
                        try:
                            energy,specRad=self.wavelengthToEV(data.getSplitData2D(xCol=1, yCol=yCol)[0], data.getSplitData2D(xCol=1,yCol=yCol)[1]- bg)
                            data.setData(Data.mergeData((data.getSplitData2D(xCol=1, yCol=yCol)[0], data.getSplitData2D(xCol=1,yCol=yCol)[1]- bg,data.getSplitData2D(xCol=1,yCol=yCol)[1]- bg,energy,specRad,specRad)))
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
    
    
    def plotDoubleGauss(self,fitter,n):
        xdata=fitter.CurveData.getSplitData2D()[0]
        ydata1=self.gauss(xdata, *fitter.params[0:3])
        ydata2=self.gauss(xdata, *fitter.params[3:6])
        textPos=fitter.textPos
        textPos2=[fitter.textPos[0],fitter.textPos[1]+0.15]
        amp=fitter.params[1]/(np.sqrt(2*np.pi*fitter.params[2]**2))
        amp2=fitter.params[1+3]/(np.sqrt(2*np.pi*fitter.params[2+3]**2))
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
        ydata1=self.gauss(xdata, *fitter.params[0:3])
        ydata2=self.gauss(xdata, *fitter.params[3:6])
        ydata3=self.gauss(xdata, *fitter.params[6:9])
        textPos=fitter.textPos
        textPos2=[fitter.textPos[0],fitter.textPos[1]+0.15]
        textPos3=[fitter.textPos[0],fitter.textPos[1]+0.3]
        amp=fitter.params[1]/(np.sqrt(2*np.pi*fitter.params[2]**2))
        amp2=fitter.params[1+3]/(np.sqrt(2*np.pi*fitter.params[2+3]**2))
        amp3=fitter.params[1+6]/(np.sqrt(2*np.pi*fitter.params[2+6]**2))
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
    
    def rect(self,x,y,w,h,c):
        polygon = matplotlib.pyplot.Rectangle((x,y),w,h,color=c)
        self.ax.add_patch(polygon)
    
    
    def rainbow_fill(self,X,Y, cmap=matplotlib.pyplot.get_cmap("nipy_spectral")):
        dx = X[1]-X[0]
        S  = 380 
        N  = 675
        h = 0.01
        if not self.rainbowMode:
            for n, (x,y) in enumerate(zip(X,Y)):
                if (x>N):
                    color= cmap(0.9999)
                elif (x<S):
                    color= cmap(0)
                else:
                    color = cmap(min((x-S)/(N-S),0.9999))
                self.rect(x,-0.035,dx,h,color)
        else:
            for n, (x,y) in enumerate(zip(X,Y)):
                if (x>N):
                    color= cmap(0.95)
                elif (x<S):
                    color= cmap(0)
                else:
                    color = cmap(min((x-S)/(N-S),0.95))
                self.rect(x-dx/2,0,dx,y,color)
    
    def xColTicksToXCol2Ticks(self, ticks):
        if self.xCol==1 and self.xCol2==4:
            ticks=ticks**-1*self.convFac
            ticks=np.around(ticks,decimals=1)
            return ["{:2.1f}".format(tick) for tick in ticks]
        elif self.xCol==4 and self.xCol2==1:
            ticks=ticks**-1*self.convFac
            ticks=np.around(ticks,decimals=0)
            return ["{:3.0f}".format(tick) for tick in ticks]
        else:
            return ticks
        
    def importData(self):
        self.dataList=[[Data(fileToNpArray(pixel, **self.fileFormat)[0]) for pixel in device] for device in self.fileList]
        return self.dataList
    
    
    
    def afterPlot(self):
        ax=self.ax
        if self.xCol==1:
            self.rainbow_fill(*self.expectData[0].getSplitData2D())
            if self.rainbowMode:
                ax.get_lines()[0].set_alpha(0)
            #ax.annotate(s="",xy=(503,0.5), xytext=(503-20,0.5), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", shrinkA=0, shrinkB=0, linewidth=mpl.rcParams["lines.linewidth"]))
            #ax.annotate(s="",xy=(520,0.5), xytext=(520+20,0.5), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", shrinkA=0, shrinkB=0, linewidth=mpl.rcParams["lines.linewidth"]))
            #bbox_props = dict(boxstyle="round,pad=0.3", fc="white", alpha=1, lw=0.5)
            #ax.text(520+30,0.55,"FWHM: 85\,meV $\equiv$ 17\,nm", bbox=bbox_props, size=self.customFontsize[2])
            #ax.annotate(s="CsPbBr\\textsubscript{3}- Emission at 511\,nm", size=self.customFontsize[2], xy=(511,0.9), xytext=(550,0.7), arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", linewidth=mpl.rcParams["lines.linewidth"]))
        #try:
        for n in range(0,len(self.expectData)):
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
        #except Exception as e:
            #warnings.warn(str(e))
                


    
    
   