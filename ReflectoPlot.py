
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


class ReflectoPlot(Plot):
    #NaturKonst
    e=1.6*10**-19 #C
    K_m=683 #lm/W
    c=2.99*10**17 #nm/s
    h=6.63*10**-34 #J*s
    #ProgKonst
    chars=list(string.ascii_uppercase) #alphabetUppercase
    convFac=(h*c)/e #eV*nm
    
    @classmethod
    def wavelengthToEV(cls,wavelength,intens):
        energy=wavelength**-1*cls.convFac #eV
        corFac=wavelength**2*cls.e/(cls.h*cls.c) #nm/eV
        intensEnergy=intens*corFac #W/(sr*m^2*eV)
        return (energy,intensEnergy)
    
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
                 fileListRefl,
                 fileListTrans,
                 fileFormat={"separator":"\t", "skiplines":0},
                 fileFormat2=None,
                 title=None,
                 showTrans=True,
                 showRefl=True,
                 showAbs=False,
                 showColAxType=["lin","lin","lin","lin","lin","lin","lin","lin","lin"],
                 showColAxLim=[None,None,None,None,None,None,None,None,None],
                 showColLabel= ["","Wavelength","Reflection", "Transmission", "Absorption", "Energy","Transmission", "Reflection","Absorption"],
                 showColLabelUnit=["","Wavelength (nm)","Transmission","Reflection","Absorption","Energy (eV)","Normalized Transmission", "Normalized Reflection","Normalized Absorption"],
                 averageMedian=False,
                 errors=False,
                 formatAbs="-",
                 formatTrans="--",
                 formatRefl=":",
                 reflAlpha = 0.3,
                 transAlpha = 0.3,
                 **kwargs
                ):
        Plot.__init__(self, name, [[trans,refl] for trans,refl in zip(fileListTrans,fileListRefl)], averageMedian=averageMedian, showColAxType=showColAxType, showColAxLim=showColAxLim, showColLabel=showColLabel, showColLabelUnit=showColLabelUnit, fileFormat=fileFormat, errors=errors, **kwargs)
        #dyn inits
        if title is None:
            self.title=name
        else:
            self.title=title
        if fileFormat2 is None:
            self.fileFormat2=fileFormat
        else:
            self.fileFormat2=fileFormat2
        self.fileListTrans=fileListTrans
        self.fileListRefl=fileListRefl
        self.labelsOrig=self.labels
        self.labels=[l +" "+self.showColLabel[self.showCol] for l in self.labels]
        self.showTrans=showTrans
        self.showRefl=showRefl
        self.showAbs=showAbs
        self.formatAbs=formatAbs
        self.formatTrans=formatTrans
        self.formatRefl=formatRefl
        self.transLs=formatTrans
        self.reflLs=formatRefl
        self.reflAlpha = reflAlpha
        self.transAlpha = transAlpha
        self.dataList=self.importData()

    @functools.lru_cache(maxsize=None)
    def importData(self):
        dataList=[]
        for reflData, transData in zip(self.fileListRefl,self.fileListTrans):
            a=Data(fileToNpArray(reflData, **self.fileFormat)[0])
            b=Data(fileToNpArray(transData, **self.fileFormat2)[0])
            x=a.getSplitData2D()[0]
            refl=a.getSplitData2D()[1]
            trans=b.getSplitData2D()[1]
            absorp=[1]*len(trans)-trans-refl
            energy,e_refl=self.wavelengthToEV(x, refl)
            e_trans=self.wavelengthToEV(x, refl)[1]
            e_absorp=[1]*len(e_trans)-e_trans-e_refl
            e_refl=self.normalize(e_refl)
            e_trans=self.normalize(e_trans)
            e_absorp=self.normalize(e_absorp)
            data=Data.mergeData([x,refl,trans,absorp,energy,e_refl,e_trans,e_absorp])
            dataList.append([Data(data)])
        return dataList

    def processFileName(self, option=".pdf"):
        if self.filename is None:
            string=self.name.replace(" ","")+self.fill+"spectra"
        else:
            string=self.filename
        if not self.scaleX is 1:
            string+=self.fill+"scaledWith{:03.0f}Pct".format(self.scaleX*100)
        return string+option
    
    def processData(self):
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
        if self.xCol==1 and self.xCol2==5:
            ticks=ticks**-1*self.convFac
            ticks=np.around(ticks,decimals=1)
            return ["{:2.1f}".format(tick) for tick in ticks]
        elif self.xCol==5 and self.xCol2==1:
            ticks=ticks**-1*self.convFac
            ticks=np.around(ticks,decimals=0)
            return ["{:3.0f}".format(tick) for tick in ticks]
        else:
            return ticks
        
    
    
    
    
    def afterPlot(self):
        ax=self.ax
        labelsOrig=self.labelsOrig
        xCol=self.xCol
        colors=self.colors
        for n in range(0,len(self.expectData)):
            if self.xCol==1:
                self.rainbow_fill(*self.expectData[0].getSplitData2D())
                if self.showTrans:
                    ax.errorbar(*self.expectData[n].getSplitData2D(xCol=xCol, yCol=2), c=colors[n], ls=self.transLs, alpha=self.transAlpha, label=labelsOrig[n]+" "+self.showColLabel[2])
            elif self.xCol==5:    
                if self.showTrans:
                    ax.errorbar(*self.expectData[n].getSplitData2D(xCol=xCol, yCol=6), c=colors[n], ls=self.transLs, alpha=self.transAlpha, label=labelsOrig[n]+" "+self.showColLabel[6])
        for n in range(0,len(self.expectData)):    
            if self.xCol==1:     
                if self.showRefl:
                    ax.errorbar(*self.expectData[n].getSplitData2D(xCol=xCol, yCol=3), c=colors[n], ls=self.reflLs, alpha=self.reflAlpha, label=labelsOrig[n]+" "+self.showColLabel[3])
            elif self.xCol==5:
                if self.showRefl:
                    ax.errorbar(*self.expectData[n].getSplitData2D(xCol=xCol, yCol=7), c=colors[n], ls=self.reflLs, alpha=self.reflAlpha, label=labelsOrig[n]+" "+self.showColLabel[7])
        for n in range(0,len(self.expectData)):
            if self.xCol==1:    
                if self.showAbs:
                    ax.errorbar(*self.expectData[n].getSplitData2D(xCol=xCol, yCol=4), c=colors[n], ls=self.ls, label=labelsOrig[n]+" "+self.showColLabel[4])
            elif self.xCol==5:
                if self.showAbs:
                    ax.errorbar(*self.expectData[n].getSplitData2D(xCol=xCol, yCol=8), c=colors[n], ls=self.ls, label=labelsOrig[n]+" "+self.showColLabel[8])
                


    
    
   