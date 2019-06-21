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





class KelvinProbePlot(Plot):

    def __init__(self,
                 name,
                 fileList,
                 fileFormat={"separator":"\t", "skiplines":42, "backoffset":0, "lastlines":0, "fileEnding":".CPD"},
                 title=None,
                 showColAxType=["lin","lin","lin","lin","lin","lin"],
                 showColAxLim=[None,None,None,None,None,None],
                 showColLabel= ["","Time","Fermi-Energy", "HOMO", "LUMO", "Gradient"],
                 showColLabelUnit=["",
                  "Time (s)",
                  "Fermi-Energy (eV)",
                  "HOMO (eV)",
                  "LUMO (eV)",
                  "Gradient"
                 ],
                 averageMedian=False,
                 errors=False,
                 probe_workfunction10=-4.25,
                 probe_workfunction05=-4.39,
                 concentenate_files_instead_of_avg=True,
                 bandgap=2.0,
                 **kwargs
                ):
        fileList=[[path+filename[:-4] for filename in os.listdir(path) if filename[-4:] == fileFormat["fileEnding"]] for path in fileList]
        Plot.__init__(self, name, fileList, averageMedian=averageMedian, showColAxType=showColAxType, showColAxLim=showColAxLim, showColLabel=showColLabel, showColLabelUnit=showColLabelUnit,showColLabelUnitNoTex=showColLabelUnit, fileFormat=fileFormat, errors=errors, concentenate_files_instead_of_avg=concentenate_files_instead_of_avg, **kwargs)
        #dyn inits
        if title is None:
            self.title=name
        else:
            self.title=title
        self.probe_workfunction10=probe_workfunction10
        self.probe_workfunction05=probe_workfunction05
        self.bandgap=bandgap

    def calc_E_fermi(self, cpd_val, kpwf=None):
        if kpwf is None:
            return self.probe_workfunction+cpd_val
        return kpwf+cpd_val
    
    def calc_HOMO_LUMO(self, E_fermi, bandgap=None):
        if bandgap == None:
            bandgap=self.bandgap
        return [E_fermi-bandgap/2,E_fermi+bandgap/2]
            
            
    def processFileName(self, option=".pdf"):
        if self.filename is None:
            string=self.name.replace(" ","")+self.fill+"fermi-energy"
        else:
            string=self.filename
        if not self.scaleX is 1:
            string+=self.fill+"scaledWith{:03.0f}Pct".format(self.scaleX*100)
        if self.filenamePrefix is not None:
            string=self.filenamePrefix+self.fill+string    
        return string+option
    
    def processData(self):
        if not self.dataProcessed:
            for device in self.dataList:
                for data in device:
                    data.limitData(xLim=self.xLimOrig)
                    kpwf=[]
                    for n in range(0,len(data.getData()[:,4])):
                        if data.getData()[:,4][n] <= 0.51:
                            kpwf.append(self.probe_workfunction05)
                        else:
                            kpwf.append(self.probe_workfunction10)
                    data.processData(self.calc_E_fermi, yCol=3, kpwf=np.asarray(kpwf))
                    time, E_fermi = data.getSplitData2D(xCol=2,yCol=3)
                    HOMO, LUMO = self.calc_HOMO_LUMO(E_fermi)
                    gradient = data.getSplitData2D(yCol=4)[1]
                    data.setData(Data.mergeData((time, E_fermi , HOMO, LUMO,gradient)))
            self.dataProcessed=True
        return self.dataList
    