
# coding: utf-8

# In[ ]:


import os
from multiprocessing import Pool


# In[ ]:



class MultiPlot:
    
    def __init__(self, plotList):
        self.plotList= plotList
        self.readyPlotList=[]
        self.readyPlotList2=[]
        self.files = []
        self.pool = Pool(4)
    
    def processPlotPair(self,plotpair, pos):
        a=[plot.doPlotObj() for plot in plotpair]
        self.readyPlotList[pos]=a
        return a
    
    def getFiles(self):
        if self.files == []:
            self.readyPlotList2=self.pool.imap(self.processPlotPair, self.plotList)
            self.files=[[plot.processFileName(option=".pdf") for plot in plotpair] for plotpair in self.readyPlotList]
        return self.files
        

