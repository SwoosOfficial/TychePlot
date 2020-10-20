
# coding: utf-8

# In[1]:


import matplotlib as mpl
#mpl.use("pgf")
import matplotlib.pyplot
import numpy as np
import scipy as sci
import scipy.interpolate as inter
import copy
import sys
import os
import string
import warnings
import functools
from functools import wraps
import copy
import inspect
from matplotlib import rc
from Filereader import fileToNpArray
from Data import Data
from Fitter import Fitter
from Plot import Plot
import matplotlib.ticker as mtick


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
    sigma_to_FWHM=2*np.sqrt(2*np.log(2))
    
    wavelength=1
    intensity_wavelength=2
    spectralRadiance_wavelength=3
    energy=4
    spectralRadiance_energy=5
    intensity_energype=7
    
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
    def lum_interpolator(cls, lumFunc, fill_value="extrapolate", kind="cubic"):
        """
        Prepare an interpolator, that extrapolates or returns zero if the extrapolation is smaller than 0
        """
        interpolator=sci.interpolate.interp1d(*lumFunc, kind=kind, fill_value=fill_value, bounds_error=False)
        @wraps(interpolator)
        def wrapper(*args,**kwargs):
            return np.maximum(abs(interpolator(*args,**kwargs)*0),interpolator(*args,**kwargs))
        return wrapper

    @classmethod
    def normalize(cls, a, value=None, col=None):
        b=a-np.amin(a)
        if value is None:
            return b/np.amax(b, axis=0)
        try:
            temp_trial=value[0]
            c=[value]*len(b)
            values_array=np.vstack(c)
            return b/(values_array[:,col-1]-np.amin(a))
        except:
            return b/(value-np.amin(a))
    

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
    def createNGauss(cls, N, defaults=[1,1,1]):
        gauss_inp=list(defaults)*N
        def NGauss(x, *gauss_inp):
            val=0
            for n in range(0,3*N,3):
                val+=cls.gauss(x,*gauss_inp[n:n+3])
            return val
        return NGauss
    
    @classmethod
    #see https://journals.aps.org/prapplied/supplemental/10.1103/PhysRevApplied.13.024061/SI-revised_final.pdf
    def createFrankCondonGauss(cls, m, S=1, E_0=2, E_vib=0.2, amp=1, sigma=0.05):
        def NFrankCGauss(E, S, E_0, E_vib, amp, sigma):
            val=0
            for n in range(0,m):
                val+=(S**n)/(np.math.factorial(n))*np.exp(-S)*cls.gauss(E,E_0-n*E_vib, amp, sigma)
            return val
        return NFrankCGauss
    
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
                 showColLabel= ["","Wavelength","Normalised Intensity", "Spectral Radiance", "Energy", "Normalised Intensity", "reduced Intensity"],
                 showColLabelUnit=["",
                  "Wavelength (nm)",
                  "Normalised Intensity",
                  "Spectral Radiance ($\\tfrac{\\mathrm{W}}{\\mathrm{sr}\\cdot \\mathrm{m}^2\\cdot \\mathrm{nm}}$)",                 
                  "Energy (eV)",
                  "Normalised Intensity", #energy scaled
                  "Reduced Intensity ($\\tfrac{1}{\\mathrm{eV}}$)"                 
                 ],
                 averageMedian=False,
                 errors=False,
                 xParamPos=0,
                 FWHMParamPos=2,
                 rainbowMode=False,
                 fitColors=['#1f77b4','#d62728','#2ca02c','#9467bd','#8c564b','#e377c2','#7f7f7f','#ff7f0e','#bcbd22','#17becf','#f8e520'],
                 bgfile=None,
                 validYTable=None,
                 normalizeMode="single",
                 showFWHM=False,
                 ticklabelformat="plain",
                 rb_pos=-0.035,
                 **kwargs
                ):
        if rainbowMode:
            kwargs.update({"legendBool":False})
        Plot.__init__(self, name, fileList, averageMedian=averageMedian, showColAxType=showColAxType, showColAxLim=showColAxLim, showColLabel=showColLabel, showColLabelUnit=showColLabelUnit, fileFormat=fileFormat, errors=errors, fitColors=fitColors, partialFitLabels=["Partial mono-Gaussian fit"], **kwargs)
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
        self.validYTable=validYTable
        self.normalizeMode=normalizeMode
        self.showFWHM=showFWHM
        self.FWHMParamPos=FWHMParamPos
        self.postprocess_normalization=False
        self.ticklabelformat=ticklabelformat
        self.rb_pos=rb_pos
        #self.dataList=self.importData()
   
    def processFileName(self, option=".pdf"):
        if self.filename is None:
            string=self.name.replace(" ","")+self.fill+"spectra"
        else:
            string=self.filename    
        if self.xCol == SpectraPlot.wavelength:
            string+=self.fill+self.showColLabel[self.showCol].replace(" ","")
        else:
            string+=self.fill+self.showColLabel[self.showCol].replace(" ","")+"vs"+self.showColLabel[self.xCol].replace(" ","") 
        if not self.scaleX == 1:
            string+=self.fill+"scaledWith{:03.0f}Pct".format(self.scaleX*100)
        if self.filenamePrefix is not None:
            self.processFileName_makedirs()
            string=self.filenamePrefix+string
        if self.normalizeMode is None:
            string+=self.fill+"not"
            string+=self.fill+"normalised"
        if self.normalizeMode == "global":
            string+=self.fill+"globally"
            string+=self.fill+"normalised"
        if self.rainbowMode:
            string+=self.fill+"rainbow"
        return string+option
    
    def process_normalization(self, data):
        if self.normalizeMode == 'single':
            data.processData(self.normalize, yCol=2)
            data.processData(self.normalize, yCol=5)
            data.processData(self.normalize, yCol=6)
            self.postprocess_normalization=False
        elif self.normalizeMode == 'global':
            self.postprocess_normalization=True
        else:
            return
    
    def __sub_processData(self, data, yCol, backg=None):
        try:
            if backg is None:
                raise TypeError
            energy,specRad=self.wavelengthToEV(data.getSplitData2D(xCol=1, yCol=yCol)[0], data.getSplitData2D(xCol=1,yCol=yCol)[1]- data.getSplitData2D(xCol=1,yCol=backg)[1])
            specRadpe=specRad/energy
            data.setData(Data.mergeData((data.getSplitData2D(xCol=1, yCol=yCol)[0], data.getSplitData2D(xCol=1,yCol=yCol)[1]- data.getSplitData2D(xCol=1,yCol=backg)[1],data.getSplitData2D(xCol=1,yCol=yCol)[1]- data.getSplitData2D(xCol=1,yCol=backg)[1],energy,specRad,specRadpe)))
        except (IndexError,TypeError):
            energy,specRad=self.wavelengthToEV(*data.getSplitData2D(xCol=1, yCol=yCol))
            specRadpe=specRad/energy
            data.setData(Data.mergeData((data.getSplitData2D(xCol=1, yCol=yCol)[0],data.getSplitData2D(xCol=1,yCol=yCol)[1],data.getSplitData2D(xCol=1,yCol=yCol)[1],energy,specRad,specRadpe)))
        data.processData(self.noNegatives, yCol=2)
        data.processData(self.noNegatives, yCol=3)
        data.processData(self.noNegatives, yCol=5)
        data.processData(self.noNegatives, yCol=6)
        self.process_normalization(data)
        data.limitData(xLim=self.xLimOrig)
        self.dataProcessed=True
        return
    
    def processData(self):
        if not self.dataProcessed:
            if self.validYTable is not None:
                if self.bgfile is None:
                    for device, validYCol in zip(self.dataList, self.validYTable):
                        for data, yCol, bg in zip(device, validYCol, self.bgYCol):
                            self.__sub_processData(data, yCol, backg=bg) 

                else:
                    bg=fileToNpArray(self.bgfile, **self.fileFormat)[0][:,1]
                    for device, validYCol in zip(self.dataList, self.validYTable):
                        for data, yCol in zip(device,validYCol):
                            self.__sub_processData(data, yCol, backg=bg)
            else:
                if self.bgfile is None:
                    for device in self.dataList:
                        while len(self.validYCol)>len(device):
                            device.append(copy.deepcopy(device[-1]))
                        while len(self.validYCol)>len(self.bgYCol):
                            self.bgYCol.append(self.bgYCol[-1])
                        for data, yCol, bg in zip(device, self.validYCol, self.bgYCol):
                            self.__sub_processData(data, yCol, backg=bg)
                else:
                    bg=fileToNpArray(self.bgfile, **self.fileFormat)[0][:,1]
                    for device in self.dataList:
                        while len(self.validYCol)>len(device):
                            device.append(copy.deepcopy(device[-1]))
                        for data, yCol in zip(device,self.validYCol):
                            self.__sub_processData(data, yCol, backg=bg)
        
                if self.postprocess_normalization:
                    maximum_values=[]
                    for device in self.dataList:
                        for data in device:
                            maximum_values.append(data.getExtremValues(typus="max"))
                    value=np.amax(np.asarray(maximum_values), axis=0)
                    for device in self.dataList:
                        for data in device:
                            data.processData(self.normalize, yCol=2, value=value, col=2)
                            data.processData(self.normalize, yCol=5, value=value, col=5)
                            data.processData(self.normalize, yCol=6, value=value, col=6)
        return self.dataList
    
    
    def process_annotation(self, param_pos, tp, n, data, fitter, yoffset_fac=0.1, fstring="Emission at \n{:3.0f}\\,nm / {:3.2f}\\,eV", fwhm_string="Peak: {:3.0f}\\,nm / {:3.2f}\\,eV\nFWHM: {:3.0f}\\,nm / {:3.0f}\\,", override_xpos=None):
        if override_xpos is not None:
            peak=override_xpos
            FWHMval=fitter.params[self.FWHMParamPos]
        else:
            peak=fitter.params[self.xParamPos+param_pos]
            FWHMval=abs(fitter.params[self.FWHMParamPos+param_pos])
        if self.showFWHM:
            FWHM=self.sigma_to_FWHM*FWHMval
            if FWHM < 1:
                fstring=fwhm_string+"meV"
                ev_FWHM=np.round(FWHM*1000,decimals=0)
            else:
                fstring=fwhm_string+"eV"
                ev_FWHM=np.round(FWHM,decimals=2)
            FWHM_nm=self.convFac*(1/(peak-FWHM/2)-1/(peak+FWHM/2))
            se=fstring.format(np.round(self.convFac/peak,decimals=0),
                              np.round(peak,decimals=2),
                              np.round(FWHM_nm,decimals=0),
                              ev_FWHM)
        else:
            se=fstring.format(np.round(self.convFac/peak,decimals=0),
                              np.round(peak,decimals=2))
        return self.ax.annotate(s=se, size=self.customFontsize[2], xy=(peak,np.amax(data)-yoffset_fac*np.amax(data)), xytext=tp, arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", facecolor=self.fitColors[n+1+param_pos//3], edgecolor=self.fitColors[n+1+param_pos//3], linewidth=mpl.rcParams["lines.linewidth"]))
            
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
        self.process_annotation(0,tp1,n,ydata1,fitter)
        self.ax.errorbar(xdata, ydata2, c=self.fitColors[n+2], ls=self.fitLs, label="Partial mono-Gaussian fit", alpha=self.fitAlpha)
        self.process_annotation(3,tp2,n,ydata2,fitter)
        
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
        self.process_annotation(0,tp1,n,ydata1,fitter)
        #fit2
        self.ax.errorbar(xdata, ydata2, c=self.fitColors[n+2], ls=self.fitLs, label="Partial mono-Gaussian fit", alpha=self.fitAlpha)
        self.process_annotation(3,tp2,n,ydata2,fitter)
        #fit3
        self.ax.errorbar(xdata, ydata3, c=self.fitColors[n+3], ls=self.fitLs, label="Partial mono-Gaussian fit", alpha=self.fitAlpha)
        self.process_annotation(6,tp3,n,ydata3,fitter)
        
    
    def plotNFrankCGauss(self,fitter,n,amp_thresh=0.1):
        xdata=fitter.CurveData.getSplitData2D()[0]
        ydataList=[]
        self.FWHMParamPos=4
        k=0
        actual_amp=1
        S,E_0,E_vib,amp,sigma = fitter.params
        while actual_amp > amp_thresh:
            pre_fac=(S**k)/(np.math.factorial(k))*np.exp(-S)
            actual_amp=amp*pre_fac
            ydataList.append(self.gauss(xdata,E_0-k*E_vib, actual_amp, sigma))
            k+=1
        m=1
        for ydata in ydataList:
            self.ax.errorbar(xdata, ydata, c=self.fitColors[n+m], ls=self.fitLs, label=f"Order {m} transition gaussian fit", alpha=self.fitAlpha)
            self.process_annotation((m-1)*3,[fitter.textPos[0],fitter.textPos[1]-(m-1)*0.15],n,ydata,fitter, override_xpos=E_0-(m-1)*E_vib)
            m+=1
            
        
    
    
    def rect(self,x,y,w,h,c):
        polygon = matplotlib.pyplot.Rectangle((x,y),w,h,color=c)
        self.ax.add_patch(polygon)
    
    
    def rainbow_fill(self, expectData, cmap=matplotlib.pyplot.get_cmap("nipy_spectral")):
        X,Y = expectData[0].getSplitData2D()
        dx = X[1]-X[0]
        S  = 380 
        N  = 670
        if self.normalizeMode == "single" or self.normalizeMode == "global":
            h = 0.01
            p = -0.035
        else:
            maxi=np.amax(np.asarray([np.amax(expect.getSplitData2D(yCol=self.showCol)[1]) for expect in expectData]))
            h = 0.01*maxi
            p = self.rb_pos*maxi
        if not self.rainbowMode:
            for n, (x,y) in enumerate(zip(X,Y)):
                if (x>N):
                    color= cmap(0.9999)
                elif (x<S):
                    color= cmap(0)
                else:
                    color = cmap(min((x-S)/(N-S),0.9999))
                self.rect(x,p,dx,h,color)
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
        
    #def importData(self):
    #    self.dataList=[[Data(fileToNpArray(pixel, **self.fileFormat)[0], desc=fileToNpArray(pixel, **self.fileFormat)[1]) for pixel in device] for device in self.fileList]
    #    return self.dataList
    
    
    
    def afterPlot(self):
        ax=self.ax
        form=mtick.ScalarFormatter(useMathText=True)
        #form.set_scientific(True)
        form.set_powerlimits((-1, 1)) 
        ax.yaxis.set_major_formatter(form)
        ax.yaxis.get_offset_text().set_visible(False)
        if self.xCol==1:
            self.rainbow_fill(self.expectData)
            if self.rainbowMode:
                ax.get_lines()[0].set_alpha(0)
            #ax.annotate(s="",xy=(503,0.5), xytext=(503-20,0.5), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", shrinkA=0, shrinkB=0, linewidth=mpl.rcParams["lines.linewidth"]))
            #ax.annotate(s="",xy=(520,0.5), xytext=(520+20,0.5), arrowprops=dict(arrowstyle="->", connectionstyle="arc3", shrinkA=0, shrinkB=0, linewidth=mpl.rcParams["lines.linewidth"]))
            #bbox_props = dict(boxstyle="round,pad=0.3", fc="white", alpha=1, lw=0.5)
            #ax.text(520+30,0.55,"FWHM: 85\,meV $\equiv$ 17\,nm", bbox=bbox_props, size=self.customFontsize[2])
            #ax.annotate(s="CsPbBr\\textsubscript{3}- Emission at 511\,nm", size=self.customFontsize[2], xy=(511,0.9), xytext=(550,0.7), arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", linewidth=mpl.rcParams["lines.linewidth"]))
        #try:
        for n in range(0,len(self.expectData)):
            try:
                if self.fitterList[n] is not None:
                    if type(self.fitterList[n]) is list:
                        for fitter in self.fitterList[n]:
                            if fitter.function == self.doubleGauss:
                                self.plotDoubleGauss(fitter,n)
                            if fitter.function == self.tripleGauss:
                                self.plotTripleGauss(fitter,n)
                            if fitter.function.__name__ == 'NFrankCGauss':
                                print("inside")
                                self.plotNFrankCGauss(fitter,n)
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
                        if self.fitterList[n].function.__name__ == 'NFrankCGauss':
                            self.plotNFrankCGauss(self.fitterList[n],n)
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
            except AttributeError as e:
                print(e)
            except Exception as e:
                raise
                print(e)
                


    
    
   