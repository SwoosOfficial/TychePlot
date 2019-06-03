
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
    fileFormat={"separator":"\t", "skiplines":1}
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
    def candToLumen(cls, cand):
        return cand*2*np.pi
    
    @classmethod
    def calcLumEffic(cls, u, dens, cand):
        return cls.div0(cls.candToLumen(cand),(u*dens)) #lm/W
    
    @classmethod
    def calcCurEffic(cls, dens, cand):
        return cand/(dens*10)
    
    @classmethod
    def doubleLogSlope(cls, volt, dens, V_bi=0):
        oldDict=np.seterr(all="ignore")
        V=np.log(volt-V_bi)
        j=np.log(dens)
        deriv=np.gradient(j,V)
        np.seterr(**oldDict)
        return deriv
    
    @classmethod
    def semiLogSlope(cls, volt, dens):
        oldDict=np.seterr(all="ignore")
        j=np.log(dens)
        deriv=np.gradient(j,volt)
        np.seterr(**oldDict)
        return deriv
    
    @classmethod
    def generateFileList(cls, prefix, pixels=4, samples=4, fill="_", alphaOffset=0, truthTable=None):
        generatedList=[[prefix+fill+cls.chars[sample+alphaOffset]+fill+str(pixel+1) for pixel in range(0,pixels)] for sample in range(0,samples)]#
        if truthTable is None:
            return generatedList
        return [[sample for truth,sample in zip(truthTableForSample,sampleList) if truth] for truthTableForSample,sampleList in zip(truthTable,generatedList)]
        
    
    def __init__(self,
                 name,
                 fileList,
                 spectraFile=None,
                 fileFormat={"separator":"\t", "skiplines":1, "fileEnding":".uil"},
                 title=None,
                 pixelsize_mm2=3.8,
                 skipSweepBack=True,
                 noSweepBackMeasured=False,
                 averageSweepBack=False,
                 specYCol=2,
                 diodYCol=2,
                 averageMedian=True,
                 specBg=None, #background column of Jeti spectra 
                 spectraRange=(390,790,401),
                 showColAxType=["lin","lin","log","log","log","log","lin","lin","lin","lin"],
                 showColAxLim=[None,None,None,None,None,None,None,None,None,None],
                 showColLabel= ["","Voltage","Current","Current Density", "Luminance", "Radiance","Current Efficiency","Luminous Efficacy","EQE", "Exponent"],
                 showColLabelUnitNoTex=["",
                  "Voltage (V)",
                  "Current (A)",
                  "Current Density (mA/cm^2)",
                  "Luminance (cd/m^2)",
                  "Radiance (W/(sr*m^2))",
                  "Current Efficiency (cd/A)",
                  "Luminous Efficacy (lm/W)",
                  "EQE (%)",
                  "Exponent"
                 ],
                 showColLabelUnit=["",
                  "Voltage (V)",
                  "Current (A)",
                  "Current Density ($\\tfrac{\\mathsf{mA}}{\\mathsf{cm}^\\mathsf{2}}$)",
                  "Luminance ($\\tfrac{\\mathsf{cd}}{\\mathsf{m}^\\mathsf{2}}$)",
                  "Radiance ($\\tfrac{\\mathsf{W}{\\mathsf{sr}\\cdot\\mathsf{m}^\\mathsf{2}}$)",
                  "Current Efficiency ($\\tfrac{\\mathsf{cd}}{\\mathsf{A}}$)",
                  "Luminous Efficacy ($\\tfrac{\\mathsf{lm}}{\\mathsf{W}}$)",
                  "EQE (\\%)",
                  "Exponent"
                 ],
                 photodiodeFunctionFile=os.path.dirname(os.path.abspath(inspect.getsourcefile(Data)))+"/luminFunction.csv",
                 spectralDataFormat={"separator":";", "skiplines":82}, #jeti csv format
                 diodeDataFormat={"separator":",", "skiplines":0, "lastlines":50}, #custom csv format
                 titleForm="\\textbf{{{} characteristic curve of}}\n\\textbf{{the {} OLED}}",
                 legLoc=2,
                 samples=None,
                 idealDevice=-1,
                 maxEqe=5,
                 darkCurrent=None,
                 darkCurrentValidPoints=(1,20),
                 measErrors={"I_p":5*10**-12,"r":5*10**-3,"s_max":0.01608,"A_pixel":10**-7},
                 V_bi=0,
                 curIdeal=False,
                 lumThresh=0.1,
                 spec_bg_file=None,
                 **kwargs
                ):
        Plot.__init__(self, name, fileList, averageMedian=averageMedian, showColAxType=showColAxType, showColAxLim=showColAxLim, showColLabel=showColLabel, showColLabelUnit=showColLabelUnit, showColLabelUnitNoTex=showColLabelUnitNoTex,fileFormat=fileFormat, legLoc=legLoc, **kwargs)
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
            self.title=title
        if spectraFile is None or spectraFile == "":
            warnings.warn("No SpectraFile given, Radiance and EQE will be wrong!")
        if type(spectraFile) is not list and type(spectraFile) is not tuple:
            self.spectraFiles=(spectraFile,)
        else:
            self.spectraFiles=spectraFile
        self.spec_bg_file=spec_bg_file
        self.specBg=specBg
        self.specYCol=specYCol
        self.photodiodeFunctionFile=photodiodeFunctionFile
        if self.overrideFileList:
            self.samples=samples
        else:
            self.samples=len(self.fileList)
        self.idealDevice=idealDevice
        self.averageSweepBack=averageSweepBack
        self.maxEqe=maxEqe
        self.darkCurrent=darkCurrent
        self.darkCurrentValidPoints=darkCurrentValidPoints
        self.measErrors=measErrors
        self.V_bi=V_bi
        self.curIdeal=curIdeal
        self.lumThresh=lumThresh
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
                if self.spec_bg_file is None:
                    bgData=spectralData.getSplitData2D(yCol=bg)[1]
                else:
                    bgData=Data(fileToNpArray(spectraFile, **self.spectralDataFormat)[0]).getSplitData2D()[1]
                yDataCor=spectralData.getSplitData2D(yCol=yCol)[1]-bgData
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
    
    def remDarkCurr(self,phot):
        if self.darkCurrent is None:
            a= phot-np.average(phot[self.darkCurrentValidPoints[0]:self.darkCurrentValidPoints[1]])
        else:
            a= phot-self.darkCurrent
        return np.maximum(a*0+10**-14,np.absolute(a))
    
    def radToCandela(self, rad, spectralData):
        summe=np.sum([self.diodeData.getSplitData2D()[1][a]*spectralData.getSplitData2D()[1][a] for a in range(0,len(self.diodeData.getSplitData2D()[1]))])
        return rad*OLEDPlot.K_m*summe
    
    def curToDensity(self,cur):
        return cur*10**5/self.pixelsize_mm2; #converts A to mA/cm² 
    
    def densToCur(self,dens):
        return dens*self.pixelsize_mm2/10**5; #converts mA/cm² to A
    
    def photToCandela(self,phot):
        return phot*4.3*self.pixelsize_mm2**(-1)*10**10 #converts A to cd/m² Correction: pixelsize_mm2
    
    def candToPhotoCurr(self, cand):
        return cand/(4.3*self.pixelsize_mm2**(-1)*10**10)
    
    def candToRadiance(self,cand, spectralData): #converts cd/m² to W/(sr*m²)
        summe=np.sum([self.diodeData.getSplitData2D()[1][a]*spectralData.getSplitData2D()[1][a] for a in range(0,len(self.diodeData.getSplitData2D()[1]))])
        return cand/(OLEDPlot.K_m*summe)
    
    def calcEQE(self, dens, rad, spectralData):
        sum2=np.sum([spectralData.getSplitData2D()[1][a]/spectralData.getSplitData2D()[0][a] for a in range(0,len(spectralData.getSplitData2D()[1]))])
        return (np.pi*rad*OLEDPlot.e)/(dens*10*OLEDPlot.h*OLEDPlot.c*sum2)*100 #dens*10**4--> mA/m² --> dens*10 --> A/m²
    
    def theoLimitPhot(self, volt ,dens, spectralData):
        sum2=np.sum([spectralData.getSplitData2D()[1][a]/spectralData.getSplitData2D()[0][a] for a in range(0,len(spectralData.getSplitData2D()[1]))])
        for n in range(0,len(volt)):
            if volt[n]>=0:
                break
        dens[0:n]=0
        rad=(self.maxEqe*dens*10*OLEDPlot.h*OLEDPlot.c*sum2)/(np.pi*OLEDPlot.e*100)
        return self.candToPhotoCurr(self.radToCandela(rad, spectralData))
    
    def theoLimitCur(self, cand, spectralData):
        sum2=np.sum([spectralData.getSplitData2D()[1][a]/spectralData.getSplitData2D()[0][a] for a in range(0,len(spectralData.getSplitData2D()[1]))])
        rad=self.candToRadiance(cand, spectralData)
        for n in range(0,len(cand)):
            if cand[n]>=self.lumThresh:
                break
        rad=rad[n:]
        dens=(np.pi*rad*OLEDPlot.e)/(self.maxEqe*10*OLEDPlot.h*OLEDPlot.c*sum2)*100
        return self.densToCur(dens)
    
    def addMeasError(self,data):
        #Volterr
        currErr=np.absolute(self.measError["I_p"]+data[1]*0)
        densErr=np.sqrt((self.curToDensity(data[2])/self.pixelsize_mm2*self.measError["A_pixel"])**2
        +(self.curToDensity(self.self.measError["I_p"]+data[2]*0))**2)
        lumErr=np.sqrt((self.photToCandela(1)*self.measError["I_p"]+data[3]*0)**2+(data[3]*self.measError["r"]/1)**2)
        photErr=self.measError["I_p"]
        #wip
    
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
        if not self.filenamePrefix is None:
            string=self.filenamePrefix+self.fill+string
        return string+option

    def processData(self):
        if not self.dataProcessed:
            for deviceData,spectralData in zip(self.dataList,self.spectralDataList):
                nList=[]
                mList=[]
                for data in deviceData:
                    data.limitData(xLim=self.xLimOrig, xCol=self.xColOrig)
                    data.processData(self.remDarkCurr, yCol=3)
                    data.processData(OLEDPlot.absolute, yCol=2)
                    if self.averageSweepBack and not self.noSweepBackMeasured:
                        array=data.getData()
                        array1=array[:int(len(array))//2]
                        array2=array[int(len(array))//2:]
                        array2=array2[::-1]
                        array=[[a,b] for a,b in zip(array1,array2)]
                        result=np.average(array, axis=1)
                        data.setData(result)
                    subDataList=[]
                    volt=data.getSplitData2D(xCol=1, yCol=2)[0]
                    subDataList.append(volt)
                    subDataList.append(data.getSplitData2D(xCol=1, yCol=2)[1]) #Current [2]  
                    dens=Data.processDataAndReturnArray(data, self.curToDensity)[:,1]
                    subDataList.append(dens) #Current_density [3]
                    lum=Data.processDataAndReturnArray(data, self.photToCandela, yCol=3)[:,2]
                    subDataList.append(lum) #Luminance [4]
                    rad=self.candToRadiance(lum,spectralData)
                    subDataList.append(rad) #Radiance [5]
                    curEffic=OLEDPlot.calcCurEffic(dens,lum)
                    subDataList.append(curEffic) #Current_efficacy
                    lumEffic=OLEDPlot.calcLumEffic(subDataList[0],dens,lum)
                    subDataList.append(lumEffic) #Luminous_efficacy
                    eqe=self.calcEQE(dens,rad,spectralData)
                    subDataList.append(eqe) #EQE8
                    power=self.doubleLogSlope(np.abs(volt-self.V_bi),np.abs(dens))
                    subDataList.append(power) #power9
                    data.setData(Data.mergeData(subDataList))
                    if self.xLim is not None:
                        try:
                            if self.limCol is None:
                                nList.append(data.getFirstIndexWhereGreaterOrEq(self.xCol,self.xLim[0]))
                                mList.append(data.getLastIndexWhereSmallerOrEq(self.xCol,self.xLim[1]))
                            else:
                                nList.append(data.getFirstIndexWhereGreaterOrEq(self.limCol,self.xLim[0]))
                                mList.append(data.getLastIndexWhereSmallerOrEq(self.limCol,self.xLim[1]))
                        except IndexError as ie:#
                            warnings.warn("Invalid Limits at column "+str(ie)[-1:]+" with value "+str(ie)[45:52])
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
                data.processData(self.remDarkCurr, yCol=3)
                data.processData(OLEDPlot.absolute, yCol=2)
                if self.averageSweepBack and not self.noSweepBackMeasured:
                    array=data.getData()
                    array1=array[:int(len(array))//2]
                    array2=array[int(len(array))//2:]
                    array2=array2[::-1]
                    array=[[a,b] for a,b in zip(array1,array2)]
                    result=np.average(array, axis=1)
                    data.setData(result)
                subDataList=[]
                #print(data.getData())
                volt=data.getSplitData2D(xCol=1, yCol=2)[0]
                subDataList.append(volt)
                subDataList.append(data.getSplitData2D(xCol=1, yCol=2)[1]) #Current [2]  
                dens=Data.processDataAndReturnArray(data, self.curToDensity)[:,1]
                subDataList.append(dens) #Current_density [3]
                lum=Data.processDataAndReturnArray(data, self.photToCandela, yCol=3)[:,2]
                subDataList.append(lum) #Luminance [4]
                rad=self.candToRadiance(lum,spectralData)
                subDataList.append(rad) #Radiance [5]
                curEffic=OLEDPlot.calcCurEffic(dens,lum)
                subDataList.append(curEffic) #Current_efficacy
                lumEffic=OLEDPlot.calcLumEffic(subDataList[0],dens,lum)
                subDataList.append(lumEffic) #Luminous_efficacy
                eqe=self.calcEQE(dens,rad,spectralData)
                subDataList.append(eqe) #EQE8
                power=self.doubleLogSlope(np.abs(volt-self.V_bi),np.abs(dens))
                subDataList.append(power) #power9
                data.setData(Data.mergeData(subDataList))
                if self.xLim is not None:
                    try:
                        if self.limCol is None:
                            nList.append(data.getFirstIndexWhereGreaterOrEq(self.xCol,self.xLim[0]))
                            mList.append(data.getLastIndexWhereSmallerOrEq(self.xCol,self.xLim[1]))
                        else:
                            nList.append(data.getFirstIndexWhereGreaterOrEq(self.limCol,self.xLim[0]))
                            mList.append(data.getLastIndexWhereSmallerOrEq(self.limCol,self.xLim[1]))
                    except IndexError as ie:#
                        warnings.warn("Invalid Limits at column "+str(ie)[-1:]+" with value "+str(ie)[45:52])
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
            if self.skipSweepBack and not self.averageSweepBack and not self.noSweepBackMeasured:
                self.dataList=[[Data(fileToNpArray(pixel, **self.fileFormat)[0][:int(len(fileToNpArray(pixel, **self.fileFormat)[0])//2)], xCol=self.xCol, yCol=self.showCol) for pixel in device] for device in self.fileList]

            else:
                self.dataList=[[Data(fileToNpArray(pixel, **self.fileFormat)[0], xCol=self.xCol, yCol=self.showCol) for pixel in device] for device in self.fileList]
            for dataSubList in self.dataList:
                for data in dataSubList:
                    data.removeDoubles(xCol=self.xColOrig)
                    
        return self.dataList
    
    
    def afterPlot(self):
        if self.idealDevice>=0:
            idealDevice=self.idealDevice
            spectralData=self.spectralDataList[idealDevice]
            expVolt=self.expectData[idealDevice].getData()[:,0]
            expCurr=self.expectData[idealDevice].getData()[:,1]
            expDens=self.expectData[idealDevice].getData()[:,2]
            if not self.curIdeal:
                idLum=self.theoLimitPhot(expVolt, expDens,spectralData)
                idPhot=idLum
                data=Data(Data.mergeData((expVolt,expCurr,idPhot)))
            else:
                expLum=self.expectData[idealDevice].getData()[:,3]
                idCur=self.theoLimitCur(expLum, spectralData)
                expPhot=self.candToPhotoCurr(expLum)
                expPhot=expLum[-len(idCur):]
                expVolt=expVolt[-len(idCur):]
                data=Data(Data.mergeData((expVolt,idCur,expPhot)))
            #data.processData(OLEDPlot.remDarkCurr, yCol=3)
            data.limitData(xLim=self.xLimOrig, xCol=self.xColOrig)
            data.processData(OLEDPlot.absolute, yCol=2)
            subDataList=[]
            volt=data.getSplitData2D(xCol=1,yCol=2)[0]
            subDataList.append(volt)
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
            power=self.doubleLogSlope(np.abs(volt-self.V_bi),np.abs(dens))
            subDataList.append(power) #power9
            data.setData(Data.mergeData(subDataList))
            self.idealData=data
            try:
                if not self.curIdeal:
                    if self.showCol not in [0,1,2,3,8,9]:
                        AX=self.ax.errorbar(*data.getSplitData2D(xCol=self.xCol, yCol=self.showCol), c="#000000", ls=self.ls, label=self.showColLabel[self.showCol]+" at a theo. EQE of "+str(self.maxEqe)+"\\,\\%")
                    if self.showCol2 not in [0,1,2,3,8,9]:
                        AX=self.ax2.errorbar(*data.getSplitData2D(xCol=self.xCol, yCol=self.showCol2), c="#000000", ls=self.ax2ls, label=self.showColLabel[self.showCol2]+" at a theo. EQE of "+str(self.maxEqe)+"\\,\\%")
                else:
                    if self.showCol not in [0,1,4,5,6,7,8]:
                        AX=self.ax.errorbar(*data.getSplitData2D(xCol=self.xCol, yCol=self.showCol), c="#000000", ls=self.ls, label=self.showColLabel[self.showCol]+" at a theo. EQE of "+str(self.maxEqe)+"\\,\\%")
                    if self.showCol2 not in [0,1,4,5,6,7,8]:
                        AX=self.ax2.errorbar(*data.getSplitData2D(xCol=self.xCol, yCol=self.showCol2), c="#000000", ls=self.ax2ls, label=self.showColLabel[self.showCol2]+" at a theo. EQE of "+str(self.maxEqe)+"\\,\\%")
            except:
                raise
        #for a in
            
                 
             
            

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
        return cls(obj.name,
                   obj.fileList,
                   dataList=obj.dataList,
                   errList=[obj.expectData,obj.deviaData,obj.logErr],
                   dataProcessed=True,
                   averageProcessed=True,
                   dataImported=True,
                   **kwargs
                   )
    
    def __init__(self,
                 name, #[[deviceApx1,deviceApx2],[deviceBpx1,deviceBpx2]]"
                 fileList,
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
            
