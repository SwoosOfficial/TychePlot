
# coding: utf-8

# In[26]:

import numpy as np
import Filereader
import Data

# In[2]:

class XYZData():
    
    def __init__(self, data, desc=None, xCol=1, yCol=2):
        self.__data = data[1:][:,1:]
        self.axes_vals=[data[:,0][1:],data[0][1:]]
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
        if xCol==1:
            xdata=np.asarray(self.axes_vals[xCol-1])
            ydata=np.asarray(data[:,yCol])
        elif xCol==2:
            xdata=np.asarray(self.axes_vals[xCol-1])
            ydata=np.asarray(data[yCol])
        else:
            raise Exception("Invalid xCol for 3D Data")
        return xdata, ydata
    
    
    def offsetData(self, offset):
        xdata, ydata = self.getSplitData2D()
        xdata = xdata+offset[0]
        ydata = ydata+offset[1]
        raise Exception("Not Yet Implemented")
        #self.setData(Data.mergeData2D(xdata, ydata))
        
    def modifyData(self, function, x=False, y=True, xCol=0, yCol=0, **kwargs):
        xdata, ydata = self.getSplitData2D(xCol=xCol, yCol=yCol) 
        
        if (x == True):
            kwargs.update({"ydata":ydata})
            xdata = function(xdata, **kwargs)
        if (y == True):
            kwargs.update({"xdata":xdata})
            ydata = function(ydata, **kwargs)
        raise Exception("Not Yet Implemented")
        self.setData(Data.mergeData2D(xdata, ydata))
            
    def processData(self, function, x=False, y=True, xCol=1, yCol=2, **kwargs):
        data = self.getData()
        if (x == True):
            data[range(0,len(data)),[xCol-1]*len(data)] = function(data[range(0,len(data)),[xCol-1]*len(data)], **kwargs)
        if (y == True):
            data[range(0,len(data)),[yCol-1]*len(data)] = function(data[range(0,len(data)),[yCol-1]*len(data)], **kwargs)
        self.setData(data)

    
    def limitData(self, xLim=None, yLim=None, xCol=None, yCol=None, keepLimits=True):
        if xCol is None:
                xCol=self.xCol
        if yCol is None:
                yCol=self.yCol
        if keepLimits:
            if isinstance(xLim, list) or isinstance(xLim, tuple):
                if len(xLim) == 2:
                    self.axes_vals=axes_vals[(axes_vals[:,xCol-1]>=xLim[0]) & (axes_vals[:,xCol-1]<=xLim[1])]
                    data=data[(data[:,xCol-1]>=xLim[0]) & (data[:,xCol-1]<=xLim[1])]
            if isinstance(yLim, list) or isinstance(yLim, tuple):
                if len(yLim) == 2:
                    data=data[(data[:,yCol-1]>=yLim[0]) & (data[:,yCol-1]<=yLim[1])]
        else:
            if isinstance(xLim, list) or isinstance(xLim, tuple):
                if len(xLim) == 2:
                    data=data[(data[:,xCol-1]>xLim[0]) & (data[:,xCol-1]<xLim[1])]
            if isinstance(yLim, list) or isinstance(yLim, tuple):
                if len(yLim) == 2:
                    data=data[(data[:,yCol-1]>yLim[0]) & (data[:,yCol-1]<yLim[1])]
        if list(data) == []:
            raise IndexError("Empty data, invalid limits")
        self.setData(data)
        
    def intersectData(self, intervals, column=None, keepLimits=True, invert_intervals=False):
        raise Exception("Not Yet Implemented")
        data = self.getData()
        if column is None:
            column = self.xCol
        intersec_data=[]
        for interval in intervals:
            if invert_intervals:
                interval=(interval[1],interval[0])
            rem_data=[]
            if keepLimits:
                if len(interval) == 2:
                    rem_data=data[(data[:,column-1]>=interval[0]) & (data[:,column-1]<=interval[1])]
            else:
                if len(xLim) == 2:
                    rem_data=data[(data[:,column-1]>interval[0]) & (data[:,column-1]<inteval[1])]
            intersec_data.append(rem_data)
        intersec_data_array=np.vstack(intersec_data)
        self.setData(intersec_data_array)
    
    
        
    def removeData(self, rangeZ, column, keepLimits=True):
        data = self.getData()
        raise Exception("Not Yet Implemented")
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
        raise Exception("Not Yet Implemented")
        if (feature == 0) or (feature == 1):
            a=self.getSplitData2D()[feature]
            return np.amin(a)
        else:
            return 0
        
    
    def getFirstIndexWhereGreaterOrEq(self,column,value,tolerance=0):
        raise Exception("Not Yet Implemented")
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(0,len(colVec)):
            if colVec[n]>=value-tolerance:
                return n
        raise IndexError("No Index found where a value is greater than {:7.1E}, invalid limits at column:".format(value)+repr(column))
    
    def getLastIndexWhereSmallerOrEq(self,column,value,tolerance=0):
        raise Exception("Not Yet Implemented")
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(1,len(colVec)):
            if colVec[-n]<=value-tolerance:
                return -n
        raise IndexError("No Index found where a value is smaller than {:7.1E}, invalid limits at column:".format(value)+repr(column))
    
    def getLastIndexWhereGreaterOrEq(self,column,value,tolerance=0):
        raise Exception("Not Yet Implemented")
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(1,len(colVec)):
            if colVec[-n]>=value-tolerance:
                return -n
        raise IndexError("No Index found where a value is greater than {:7.1E}, invalid limits at column:".format(value)+repr(column))
    
    def getFirstIndexWhereSmallerOrEq(self,column,value,tolerance=0):
        raise Exception("Not Yet Implemented")
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(0,len(colVec)):
            if colVec[n]<=value-tolerance:
                return n
        raise IndexError("No Index found where a value is greater than {:7.1E}, invalid limits at column:".format(value)+repr(column))
    
    def removeAllButEveryXthLine(self,factor):
        raise Exception("Not Yet Implemented")
        self.setData(self.__data[::factor])
        
    def getDataWithOnlyEveryXthLine(self, factor):
        raise Exception("Not Yet Implemented")
        return Data(self.__data[::factor])
    
    def removeDoubles(self, xCol=0):
        raise Exception("Not Yet Implemented")
        if xCol==0:
            xCol=self.xCol
        tup=np.unique(self.__data[:,xCol-1], return_index=True, axis=0)
        self.setData(self.__data[tup[1]])


