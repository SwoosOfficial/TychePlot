
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


class OLEDPlot(Plot):
    #NaturKonst
    e=1.6*10**-19
    K_m=683 #lm/W
    c=2.99*10**17 #nm/s
    h=6.63*10**-34 #J*s
    k_B=1.38064852*10**-23# J/K
    eps_0=8.85418781762*10**-12 #As/(Vm)
    #ProgKonst
    chars=list(string.ascii_uppercase) #alphabetUppercase
    
    Voltage=1
    Current=2
    Current_density=3
    Luminance=4
    Radiance=5
    Current_Efficacy=6
    Luminous_Efficacy=7
    EQE=8
    
    @classmethod
    def ohmicLaw(cls, volt, conductivity):
        return volt*conductivity
    
    @classmethod
    def shockleyEquation(cls, volt, saturationCurrent, idealityFactor, temperature):
        return saturationCurrent*(np.exp(OLEDPlot.e*volt/(cls.k_B*temperature*idealityFactor))-1)
    
    @classmethod
    def gurneyMottEquation(cls, volt, thickness, permittivity, mobility):
        return 9*permittivity*cls.eps_0*mobility*(volt)**2/(8*thickness**3)
                       
    @classmethod
    def oledjVCharFunc(cls, volt, conductivity, saturationCurrent, idealityFactor, temperature, thickness, permittivity, mobility, voltOnset, sclcOnset):
        return cls.ohmicLaw(volt,conductivity)+np.heaviside(volt-voltOnset,0)*cls.shockleyEquation(volt-voltOnset, saturationCurrent, idealityFactor, temperature)*np.heaviside(sclcOnset-volt,1)+np.heaviside(volt-sclcOnset,0)*cls.gurneyMottEquation(volt, thickness, permittivity, mobility)
    
    @classmethod
    def remDarkCurr(cls,phot):
        return phot-np.amin(phot)
    
    @classmethod
    def candToLumen(cls, cand):
        return cand*2*np.pi
    
    @classmethod
    def calcLumEffic(cls, u, dens, cand):
        return cls.div0(cls.candToLumen(cand),(u*dens)) #lm/W
    
    @classmethod
    def calcCurEffic(cls, dens, cand):
        return cand/dens
    
    def __init__(self,
                 name,
                 fileList,
                 spectraFile=None,
                 fileFormat={"separator":"\t", "skiplines":1},
                 title=None,
                 pixelsize_mm2=3.8,
                 skipSweepBack=True,
                 noSweepBackMeasured=False,
                 specYCol=2,
                 diodYCol=2,
                 averageMedian=True,
                 specBg=None, #background column of Jeti spectra 
                 spectraRange=(390,790,401),
                 showColAxType=["lin","lin","log","log","log","log","lin","lin","lin"],
                 showColAxLim=[None,None,None,None,None,None,None,None,None],
                 showColLabel= ["","Voltage","Current","Current Density", "Luminance", "Radiance","Current Efficiency","Luminous Efficacy","EQE"],
                 showColLabelUnit=["",
                  "Voltage (V)",
                  "Current (A)",
                  "Current Density ($\\tfrac{\\mathrm{mA}}{\\mathrm{cm}^2}$)",
                  "Luminance ($\\tfrac{\\mathrm{cd}}{\\mathrm{m}^2}$)",
                  "Radiance ($\\tfrac{\\mathrm{W}{\\mathrm{sr}\\cdot\\mathrm{m}^2}$)",
                  "Current Efficiency ($\\tfrac{\\mathrm{cd}}{\\mathrm{A}}$)",
                  "Luminous Efficacy ($\\tfrac{\\mathrm{lm}}{\\mathrm{W}}$)",
                  "EQE (\\%)"
                 ],
                 photodiodeFunctionFile=os.path.dirname(os.path.abspath(inspect.getsourcefile(Data)))+"/luminFunction.csv",
                 spectralDataFormat={"separator":";", "skiplines":82}, #jeti csv format
                 diodeDataFormat={"separator":",", "skiplines":0, "lastlines":50}, #custom csv format
                 titleForm="\\textbf{{{} characteristic curve of}}\n\\textbf{{the {} OLED}}",
                 legLoc=2,
                 samples=None,
                 **kwargs
                ):
        Plot.__init__(self, name, fileList, averageMedian=averageMedian, showColAxType=showColAxType, showColAxLim=showColAxLim, showColLabel=showColLabel, showColLabelUnit=showColLabelUnit, fileFormat=fileFormat, legLoc=legLoc, **kwargs)
        #dyn inits
        self.pixelsize_mm2=pixelsize_mm2
        self.skipSweepBack=skipSweepBack
        self.noSweepBackMeasured=noSweepBackMeasured
        self.diodYCol=diodYCol
        self.spectraRange=spectraRange
        self.spectralDataFormat=spectralDataFormat
        self.diodeDataFormat=diodeDataFormat
        if title is None:
            self.title=titleForm.format(self.showColLabel[self.showCol],name)
        else:
            self.tilte=title
        if spectraFile is None or spectraFile == "":
            warnings.warn("No SpectraFile given, Radiance and EQE will be wrong!")
        if type(spectraFile) is not list and type(spectraFile) is not tuple:
            self.spectraFiles=(spectraFile,)
        else:
            self.spectraFiles=spectraFile
        self.specBg=specBg
        self.specYCol=specYCol
        self.photodiodeFunctionFile=photodiodeFunctionFile
        if self.overrideFileList:
            self.samples=samples
        else:
            self.samples=len(self.fileList)
        #initmethods
        self.exportDataList=copy.deepcopy(self.dataList)
        self.spectralDataList=self.spectraDataImport()[0]
        self.diodeData=self.spectraDataImport()[1]

    
    def spectraDataImport(self):
        bg=self.specBg
        yCol=self.specYCol
        diodeFuncData=Data(fileToNpArray(self.photodiodeFunctionFile, **self.diodeDataFormat)[0])
        diodeFuncData.processData(Plot.normalize2, yCol=self.diodYCol)
        spectralDataList=[]
        for spectraFile in self.spectraFiles:
            try:
                spectralData=Data(fileToNpArray(spectraFile, **self.spectralDataFormat)[0])
            except TypeError:
                spectralData=diodeFuncData
            xData=spectralData.getSplitData2D()[0]
            yData=spectralData.getSplitData2D(yCol=yCol)[1]
            try:
                bgData=spectralData.getSplitData2D(yCol=bg)[1]
                yDataCor=spectralData.getSplitData2D(yCol=yCol)[1]-spectralData.getSplitData2D(yCol=bg)[1]
                spectralData.setData(Data.mergeData((xData,yData,bgData,yDataCor)))
                spectralData.processData(OLEDPlot.absolute, yCol=2)
                spectralData.processData(OLEDPlot.absolute, yCol=4)
                spectralData.processData(OLEDPlot.normalize, yCol=2)
                spectralData.processData(OLEDPlot.normalize, yCol=4)
            except (TypeError):
                spectralData.setData(Data.mergeData((xData,yData)))
                spectralData.processData(OLEDPlot.absolute, yCol=2)
                spectralData.processData(OLEDPlot.normalize, yCol=2)
            Plot.equalizeRanges(spectralData)
            spectralDataList.append(spectralData)
        if len(spectralDataList) == 1:
            spectralDataList=spectralDataList*self.samples
        Plot.equalizeRanges(diodeFuncData)
        return spectralDataList, diodeFuncData
    
    
    def curToDensity(self,cur):
        return cur*10**5/self.pixelsize_mm2; #converts A to mA/cm² A/m²--> 
    
    def photToCandela(self,phot):
        return phot*4.3*self.pixelsize_mm2**(-1)*10**10 #converts A to cd/m² Correction: pixelsize_mm2
    
    def candToRadiance(self,cand, spectralData): #converts cd/m² to W/(sr*m²)
        summe=np.sum([self.diodeData.getSplitData2D()[1][a]*spectralData.getSplitData2D()[1][a] for a in range(0,len(self.diodeData.getSplitData2D()[1]))])
        return cand/(OLEDPlot.K_m*summe)
    
    def calcEQE(self, dens, rad, spectralData):
        sum2=np.sum([spectralData.getSplitData2D()[1][a]/spectralData.getSplitData2D()[0][a] for a in range(0,len(spectralData.getSplitData2D()[1]))])
        return (np.pi*rad*OLEDPlot.e)/(dens*10*OLEDPlot.h*OLEDPlot.c*sum2)*100 #dens*10**4--> mA/m² --> dens*10 --> A/m²
    
    def processFileName(self, option=".pdf"):
        if self.filename is None:
            string=""
        else:
            string=self.filename
        if self.showCol2 == 0:
            if self.xCol == OLEDPlot.Voltage:
                string+=self.name.replace(" ","")+self.fill+"OLED_"+self.showColLabel[self.showCol].replace(" ","")
            else:
                string+=self.name.replace(" ","")+self.fill+"OLED_"+self.showColLabel[self.showCol].replace(" ","")+"vs"+self.showColLabel[self.xCol].replace(" ","")    
        else:
            if self.xCol == OLEDPlot.Voltage:
                string+=self.name.replace(" ","")+self.fill+"OLED_"+self.showColLabel[self.showCol].replace(" ","")+"+"+self.showColLabel[self.showCol2].replace(" ","")
            else:
                string+=self.name.replace(" ","")+self.fill+"OLED_"+self.showColLabel[self.showCol].replace(" ","")+"+"+self.showColLabel[self.showCol2].replace(" ","")+"vs"+self.showColLabel[self.xCol].replace(" ","")
        
        if not self.skipSweepBack:
            string+=self.fill+"withSweepback"
        if not self.averageMedian:
            string+=self.fill+"noMedian"
        if not True in [a[1] for a in self.errors]:
            string+=self.fill+"withoutErrors"
        if not self.scaleX is 1:
            string+=self.fill+"scaledWith{:03.0f}Pct".format(self.scaleX*100)
        return string+option

    def processData(self):
        if not self.dataProcessed:
            for deviceData,spectralData in zip(self.dataList,self.spectralDataList):
                nList=[]
                mList=[]
                for data in deviceData:
                    data.limitData(xLim=self.xLimOrig, xCol=self.xColOrig)
                    data.processData(OLEDPlot.remDarkCurr, yCol=3)
                    data.processData(OLEDPlot.absolute, yCol=2)
                    subDataList=[]
                    #print(data.getData())
                    subDataList.append(data.getSplitData2D(xCol=1, yCol=2)[0])
                    subDataList.append(data.getSplitData2D(xCol=1, yCol=2)[1]) #Current [2]  
                    dens=Data.processDataAndReturnArray(data, self.curToDensity)[:,1]
                    subDataList.append(dens) #Current_density [3]
                    lum=Data.processDataAndReturnArray(data, self.photToCandela, yCol=3)[:,2]
                    subDataList.append(lum) #Luminance [4]
                    subDataList.append(self.candToRadiance(lum,spectralData)) #Radiance [5]
                    curEffic=OLEDPlot.calcCurEffic(dens,lum)
                    subDataList.append(curEffic) #Current_efficacy
                    lumEffic=OLEDPlot.calcLumEffic(subDataList[0],dens,lum)
                    subDataList.append(lumEffic) #Luminous_efficacy
                    eqe=self.calcEQE(dens,subDataList[4],spectralData)
                    subDataList.append(eqe) #EQE8
                    data.setData(Data.mergeData(subDataList))
                    if self.xLim is not None:
                        nList.append(data.getFirstIndexWhereGreaterOrEq(1,self.xLim[0]))
                        mList.append(data.getLastIndexWhereSmallerOrEq(1,self.xLim[1]))
                    try:
                        n=max(nList)
                        m=min(mList)
                        for data in deviceData:
                            if m==-1:
                                data.setData(data.getData()[n:])
                            else:
                                data.setData(data.getData()[n:m+1])
                    except ValueError:
                            pass
        self.dataProcessed=True
        return self.dataList

    def processAllAndExport(self, **kwargs):
        localDataList=self.importData()
        for deviceData,spectralData in zip(localDataList,self.spectralDataList):
            nList=[]
            mList=[]
            for data in deviceData:
                data.limitData(xLim=self.xLimOrig, xCol=self.xColOrig)
                data.processData(OLEDPlot.remDarkCurr, yCol=3)
                data.processData(OLEDPlot.absolute, yCol=2)
                subDataList=[]
                subDataList.append(data.getSplitData2D(yCol=2)[0])
                subDataList.append(data.getSplitData2D(yCol=2)[1]) #Current [2]  
                dens=Data.processDataAndReturnArray(data, self.curToDensity)[:,1]
                subDataList.append(dens) #Current_density [3]
                lum=Data.processDataAndReturnArray(data, self.photToCandela, yCol=3)[:,2]
                subDataList.append(lum) #Luminance [4]
                subDataList.append(self.candToRadiance(lum,spectralData)) #Radiance [5]
                curEffic=OLEDPlot.calcCurEffic(dens,lum)
                subDataList.append(curEffic) #Current_efficacy
                lumEffic=OLEDPlot.calcLumEffic(subDataList[0],dens,lum)
                subDataList.append(lumEffic) #Luminous_efficacy
                eqe=self.calcEQE(dens,subDataList[4],spectralData)
                subDataList.append(eqe) #EQE8
                data.setData(Data.mergeData(subDataList))
                if self.xLim is not None:
                    nList.append(data.getFirstIndexWhereGreaterOrEq(1,self.xLim[0]))
                    mList.append(data.getLastIndexWhereSmallerOrEq(1,self.xLim[1]))
                try:
                    n=max(nList)
                    m=min(mList)
                    for data in deviceData:
                        if m==-1:
                            data.setData(data.getData()[n:])
                        else:
                            data.setData(data.getData()[n:m+1])
                except ValueError:
                    pass
        expectData, errData =self.calcCustomAverage(localDataList)
        self.exportAllData(expectData=expectData, errData=errData, **kwargs)
        return expectData, errData
            
        
    def importData(self):
        if not self.dataImported:
            if self.skipSweepBack and not self.noSweepBackMeasured:
                self.dataList=[[Data(fileToNpArray(pixel, **self.fileFormat)[0][:int(len(fileToNpArray(pixel, **self.fileFormat)[0])//2)], xCol=self.xCol, yCol=self.showCol) for pixel in device] for device in self.fileList]

            else:
                self.dataList=[[Data(fileToNpArray(pixel, **self.fileFormat)[0], xCol=self.xCol, yCol=self.showCol) for pixel in device] for device in self.fileList]
        return self.dataList
    


# In[3]:


class OLEDNamedListPlot(OLEDPlot):
    
    @classmethod
    def initByExistingPlot(cls, obj, **kwargs):
        return cls(obj.prefix,
                   obj.labels,
                   obj.name,
                   dataList=obj.dataList,
                   errList=[obj.expectData,obj.deviaData,obj.logErr],
                   dataProcessed=True,
                   averageProcessed=True,
                   dataImported=True,
                   **kwargs
                   )
    
    def __init__(self,
                 prefix,
                 labels,
                 name,
                 offset=0,
                 step=1,
                 pixels=4,
                 samples=4,
                 alphaOffset=0,
                 errors=[[True,True]]*4,
                 show=[[True,False]]*4,
                 **kwargs):
        OLEDPlot.__init__(self, name, [], overrideFileList=True, samples=samples, **kwargs)
        self.prefix=prefix
        if len(labels) is samples and len(show) is samples and len(errors) is samples:
            self.labels=labels
            self.errors=errors
            self.show=show
        else:
            print(labels)
            print(show)
            print(errors)
            raise 
        self.offset=offset
        self.step=step
        self.samples=samples
        self.pixels=pixels
        self.alphaOffset=alphaOffset
        #inits
        self.fileList=self.generateFileList()
        self.dataList=self.importData()

        
    def generateFileList(self):
        return [[self.prefix+self.fill+self.chars[sample+self.alphaOffset]+self.fill+str(pixel+1) for pixel in range(0,self.pixels)] for sample in range(0,self.samples)]
        
    


class OLEDCustomFileListPlot(OLEDPlot):
    
    @classmethod
    def initByExistingPlot(cls, obj, **kwargs):
        return cls(obj.fileList,
                   obj.name,
                   dataList=obj.dataList,
                   errList=[obj.expectData,obj.deviaData,obj.logErr],
                   dataProcessed=True,
                   averageProcessed=True,
                   dataImported=True,
                   **kwargs
                   )
    
    def __init__(self,
                 fileList, #[[deviceApx1,deviceApx2],[deviceBpx1,deviceBpx2]]"
                 name,
                 filename=None,
                 colorOffset=0,
                 **kwargs):
        OLEDPlot.__init__(self, name, fileList, filename=filename, ax2LegendLabelAddString=" Luminance", **kwargs)
        self.expectData=[]
        self.deviaData=[] 
        self.colorOffset=colorOffset
        self.dataList=self.importData()
        if filename is not None:
            self.filename=filename
            
class DUMP():
    pass
    """   
    def processData(self):
        if not self.dataProcessed:
            Voltage=OLEDPlot.Voltage
            Current=OLEDPlot.Current
            Current_density=OLEDPlot.Current_density
            Luminance=OLEDPlot.Luminance
            Radiance=OLEDPlot.Radiance
            Current_Efficacy=OLEDPlot.Current_Efficacy
            Luminous_Efficacy=OLEDPlot.Luminous_Efficacy
            EQE=OLEDPlot.EQE
            descDict={
                        Voltage:"volt",
                        Current:"cur",
                        Current_density:"dens",
                        Luminance:"lum",
                        Radiance:"rad",
                        Current_Efficacy:"curEffic",
                        Luminous_Efficacy:"lumEffic",
                        EQE:"EQE"
                      }
            xCol=self.xCol
            showCol=self.showCol
            showCol2=self.showCol2
            for deviceData,spectralData in zip(self.dataList,self.spectralDataList):
                nList=[]
                mList=[]
                for data in deviceData:
                    data.limitData(xLim=self.xLimOrig, xCol=self.xColOrig)
                    data.processData(OLEDPlot.remDarkCurr, yCol=3)
                    data.processData(OLEDPlot.absolute, yCol=2)
                    subDataDict={}
                    if showCol in (Voltage,Luminous_Efficacy) or showCol2 in (Voltage,Luminous_Efficacy) or xCol in (Voltage,Luminous_Efficacy):
                        subDataDict.update({"volt":data.getSplitData2D(xCol=1, yCol=2)[0]})
                    if showCol==Current or showCol2==Current or xCol==Current:
                        subDataDict.update({"cur":data.getSplitData2D(yCol=2)[1]}) #Current [1]
                    if showCol==Current_density or showCol2==Current_density or showCol>=Current_Efficacy or showCol2>=Current_Efficacy or xCol==Current_density:    
                        dens=Data.processDataAndReturnArray(data, self.curToDensity)[:,1]
                        subDataDict.update({"dens":dens}) #Current_density [2]
                    if showCol==Luminance or showCol2==Luminance or showCol>=Current_Efficacy or showCol2>=Current_Efficacy or xCol==Luminance:
                        lum=Data.processDataAndReturnArray(data, self.photToCandela, yCol=3)[:,2]
                        subDataDict.update({"lum":lum}) #Luminance [3]
                    if showCol==Radiance or showCol2==Radiance or showCol>=Current_Efficacy or showCol2>=Current_Efficacy or xCol==Radiance:
                        subDataDict.update({"rad":self.candToRadiance(lum, spectralData)}) #Radiance [4]
                    if showCol==Current_Efficacy or showCol2==Current_Efficacy or xCol==Current_Efficacy:
                        curEffic=OLEDPlot.calcCurEffic(subDataDict["dens"],subDataDict["lum"])
                        subDataDict.update({"curEffic":curEffic})
                    if showCol==Luminous_Efficacy or showCol2==Luminous_Efficacy or xCol==Luminous_Efficacy:
                        lumEffic=OLEDPlot.calcLumEffic(subDataDict["volt"],subDataDict["dens"],subDataDict["lum"])
                        subDataDict.update({"lumEffic":lumEffic})
                    if showCol==EQE or showCol2==EQE or xCol==EQE:
                        eqe=self.calcEQE(subDataDict["dens"],subDataDict["rad"], spectralData)
                        subDataDict.update({"EQE":eqe})
                    if showCol2<=0:
                        data.setData(Data.mergeData((subDataDict[descDict[self.xCol]],subDataDict[descDict[self.showCol]])))
                    else:
                        data.setData(Data.mergeData((subDataDict[descDict[self.xCol]],subDataDict[descDict[self.showCol]],subDataDict[descDict[self.showCol2]])))
                    if self.xLim is not None:
                        nList.append(data.getFirstIndexWhereGreaterOrEq(1,self.xLim[0]))
                        mList.append(data.getLastIndexWhereSmallerOrEq(1,self.xLim[1]))
                try:
                    n=max(nList)
                    m=min(mList)
                    for data in deviceData:
                        if m==-1:
                            data.setData(data.getData()[n:])
                        else:
                            data.setData(data.getData()[n:m+1])
                except ValueError:
                    pass
            self.xCol=1
            self.showCol=2
            self.showCol2=3
            self.dataProcessed=True
        return self.dataList
"""

