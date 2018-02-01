
# coding: utf-8

# In[26]:

import numpy as np
import Filereader


# In[2]:

class Data:
    
    def __init__(self, data, desc=None, xCol=1, yCol=2):
        self.__data = data
        self.xCol = xCol
        self.yCol = yCol
        self.desc = desc

        
    def __iter__(self):
        return self.getData()
        
    def getData(self): 
        return self.__data
    
    def setData(self, data):
        self.__data = data
        
    
    def getSplitData2D(self, xCol=0, yCol=0):
        data = self.getData()
        if xCol == 0:
            xCol=self.xCol
        if yCol == 0:
            yCol=self.yCol
        xdata=data[range(0,len(data)),[xCol-1]*len(data)]
        ydata=data[range(0,len(data)),[yCol-1]*len(data)]
        return xdata, ydata
    
    @classmethod
    def initData2D(cls, xdata, ydata):
        a=np.asarray([xdata,ydata], dtype=np.float64)
        return Data(a.transpose())
    
    @classmethod
    def mergeData2D(cls, xdata, ydata):
        a=np.asarray([xdata,ydata], dtype=np.float64)
        return a.transpose()
    
    @classmethod
    def mergeData(cls, dataTuple):
        a=np.asarray(dataTuple, dtype=np.float64)
        return a.transpose()
    
    @classmethod
    def processDataAndReturnArray(cls, data, function, x=False, y=True, xCol=1, yCol=2, **kwargs):
        data = data.getData()
        if (x == True):
            data[range(0,len(data)),[xCol-1]*len(data)] = function(data[range(0,len(data)),[xCol-1]*len(data)])
        if (y == True):
            data[range(0,len(data)),[yCol-1]*len(data)] = function(data[range(0,len(data)),[yCol-1]*len(data)])
        return data
    
    def offsetData(self, offset):
        xdata, ydata = self.getSplitData2D()
        xdata = xdata+offset[0]
        ydata = ydata+offset[1]
        self.setData(Data.mergeData2D(xdata, ydata))
        
    def modifyData(self, function, x=False, y=True, xCol=0, yCol=0, **kwargs):
        xdata, ydata = self.getSplitData2D(xCol=xCol, yCol=yCol) 
        
        if (x == True):
            kwargs.update({"ydata":ydata})
            xdata = function(xdata, **kwargs)
        if (y == True):
            kwargs.update({"xdata":xdata})
            ydata = function(ydata, **kwargs)
        self.setData(Data.mergeData2D(xdata, ydata))
            
    def processData(self, function, x=False, y=True, xCol=1, yCol=2, **kwargs):
        data = self.getData()
        if (x == True):
            data[range(0,len(data)),[xCol-1]*len(data)] = function(data[range(0,len(data)),[xCol-1]*len(data)])
        if (y == True):
            data[range(0,len(data)),[yCol-1]*len(data)] = function(data[range(0,len(data)),[yCol-1]*len(data)])
        self.setData(data)

    
    def limitData(self, xLim=None, yLim=None, xCol=None, yCol=None, keepLimits=True):
        data = self.getData()
        if xCol is None:
                xCol=self.xCol
        if yCol is None:
                yCol=self.yCol
        if keepLimits:
            if isinstance(xLim, list):
                if len(xLim) == 2:
                    data=data[(data[:,xCol-1]>=xLim[0]) & (data[:,xCol-1]<=xLim[1])]
            if isinstance(yLim, list):
                if len(yLim) == 2:
                    data=data[(data[:,yCol-1]>=yLim[0]) & (data[:,yCol-1]<=yLim[1])]
        else:
            if isinstance(xLim, list):
                if len(xLim) == 2:
                    data=data[(data[:,xCol-1]>xLim[0]) & (data[:,xCol-1]<xLim[1])]
            if isinstance(yLim, list):
                if len(yLim) == 2:
                    data=data[(data[:,yCol-1]>yLim[0]) & (data[:,yCol-1]<yLim[1])]
        if list(data) == []:
            raise IndexError("Empty data, invalid limits")
        self.setData(data)
        
        
    def removeData(self, rangeZ, column, keepLimits=True):
        data = self.getData()
        if keepLimits:
            if isinstance(xLim, list):
                if len(xLim) == 2:
                    data=data[(data[:,self.xCol-1]<=xLim[0]) & (data[:,self.xCol-1]>=xLim[1])]
            if isinstance(yLim, list):
                if len(yLim) == 2:
                    data=data[(data[:,self.yCol-1]<=yLim[0]) & (data[:,self.yCol-1]>=yLim[1])]
        else:
            if isinstance(xLim, list):
                if len(xLim) == 2:
                    data=data[(data[:,self.xCol-1]<xLim[0]) & (data[:,self.xCol-1]>xLim[1])]
            if isinstance(yLim, list):
                if len(yLim) == 2:
                    data=data[(data[:,self.yCol-1]<yLim[0]) & (data[:,self.yCol-1]>yLim[1])]
        self.setData(data)
        
    def getExtremValue(self, feature=1):
        if (feature == 0) or (feature == 1):
            a=self.getSplitData2D()[feature]
            return np.amin(a)
        else:
            return 0
        
    
    def getFirstIndexWhereGreaterOrEq(self,column,value,tolerance=0):
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(0,len(colVec)):
            if colVec[n]>=value-tolerance:
                return n
        raise IndexError("No Index found, invalid limits")
    
    def getLastIndexWhereSmallerOrEq(self,column,value,tolerance=0):
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(1,len(colVec)):
            if colVec[-n]<=value-tolerance:
                return -n
        raise IndexError("No Index found, invalid limits")
    
    def getLastIndexWhereGreaterOrEq(self,column,value,tolerance=0):
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(1,len(colVec)):
            if colVec[-n]>=value-tolerance:
                return -n
        raise IndexError("No Index found, invalid limits")
    
    def getFirstIndexWhereSmallerOrEq(self,column,value,tolerance=0):
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(0,len(colVec)):
            if colVec[n]<=value-tolerance:
                return n
        raise IndexError("No Index found, invalid limits")
    
    def removeAllButEveryXthLine(self,factor):
        self.setData(self.__data[::factor])
        
    def getDataWithOnlyEveryXthLine(self, factor):
        return Data(self.__data[::factor])


