
# coding: utf-8

# In[26]:

import numpy as np
import Filereader
import copy
import warnings

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
        xdata=np.asarray(data[:,xCol-1])
        ydata=np.asarray(data[:,yCol-1])
        return xdata, ydata
    
    def setSplitData2D(self, xy_tup, xCol=0, yCol=0):
        #data = self.getData()
        if xCol == 0:
            xCol=self.xCol
        if yCol == 0:
            yCol=self.yCol
        xdata=xy_tup[0]
        ydata=xy_tup[1]
        self.__data[:,xCol-1]=xdata
        self.__data[:,yCol-1]=ydata
    
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
        data = copy.deepcopy(data.getData())
        if (x == True):
            data[range(0,len(data)),[xCol-1]*len(data)] = function(data[range(0,len(data)),[xCol-1]*len(data)], **kwargs)
        if (y == True):
            data[range(0,len(data)),[yCol-1]*len(data)] = function(data[range(0,len(data)),[yCol-1]*len(data)], **kwargs)
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
            data[range(0,len(data)),[xCol-1]*len(data)] = function(data[range(0,len(data)),[xCol-1]*len(data)], **kwargs)
        if (y == True):
            data[range(0,len(data)),[yCol-1]*len(data)] = function(data[range(0,len(data)),[yCol-1]*len(data)], **kwargs)
        self.setData(data)

    
    def limitData(self, xLim=None, yLim=None, xCol=None, yCol=None, keepLimits=True, check_seq=5, legacy=False):
        data = self.getData()
        if xCol is None:
            xCol=self.xCol
        if yCol is None:
            yCol=self.yCol
        if legacy:
            if keepLimits:
                if isinstance(xLim, list) or isinstance(xLim, tuple):
                    if len(xLim) == 2:
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
        else:
            if keepLimits:
                if isinstance(xLim, list) or isinstance(xLim, tuple):
                    if len(xLim) == 2:
                        try:
                            start=self.getFirstIndexWhereGreaterOrEq(xCol, xLim[0], check_seq=check_seq)
                        except IndexError:
                            start=0
                            warnings.warn("Invalid start limit")
                        try:
                            end=self.getFirstIndexWhereGreaterOrEq(xCol, xLim[1], check_seq=check_seq)
                        except IndexError:
                            end=len(data)
                            warnings.warn("Invalid end limit")
                        try:
                            data=data[start:end+1]
                        except IndexError:
                            data=data[start:end]
                if isinstance(yLim, list) or isinstance(yLim, tuple):
                    if len(yLim) == 2:
                        try:
                            start=self.getFirstIndexWhereGreaterOrEq(xCol, xLim[0], check_seq=check_seq)
                        except IndexError:
                            start=0
                            warnings.warn("Invalid start limit")
                        try:
                            end=self.getFirstIndexWhereGreaterOrEq(xCol, xLim[1], check_seq=check_seq)
                        except IndexError:
                            end=len(data)
                            warnings.warn("Invalid end limit")
                        try:
                            data=data[start:end+1]
                        except IndexError:
                            data=data[start:end]
            else:
                if isinstance(xLim, list) or isinstance(xLim, tuple):
                    if len(xLim) == 2:
                        try:
                            start=self.getFirstIndexWhereGreaterOrEq(xCol, xLim[0], check_seq=check_seq)
                        except IndexError:
                            start=0
                            warnings.warn("Invalid start limit")
                        try:
                            end=self.getFirstIndexWhereGreaterOrEq(xCol, xLim[1], check_seq=check_seq)
                        except IndexError:
                            end=len(data)
                            warnings.warn("Invalid end limit")
                        data=data[start:end]   
                if isinstance(yLim, list) or isinstance(yLim, tuple):
                    if len(yLim) == 2:
                        try:
                            start=self.getFirstIndexWhereGreaterOrEq(xCol, xLim[0], check_seq=check_seq)
                        except IndexError:
                            start=0
                            warnings.warn("Invalid start limit")
                        try:
                            end=self.getFirstIndexWhereGreaterOrEq(xCol, xLim[1], check_seq=check_seq)
                        except IndexError:
                            end=len(data)
                            warnings.warn("Invalid end limit")
                        data=data[start:end]
                
        self.setData(data)
        
    def intersectData(self, intervals, column=None, keepLimits=True, invert_intervals=False):
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
        
    def getExtremValue(self, feature=1, typus="min"):
        a=self.__data[:,feature+1]
        if typus=="min":
            return np.amin(a)
        elif typus=="max":
            return np.amax(a)
        else:
            raise NotYetImplementedError
        
    def getExtremValues(self, typus="min"):
        result=[]
        for feature in range(0, len(self.__data[0])):
            a=self.getSplitData2D(yCol=feature+1)[1]
            if typus=="min":
                result.append(np.amin(a))
            elif typus=="max":
                result.append(np.amax(a))
            else:
                raise NotYetImplementedError
        return result
        
    
    def getFirstIndexWhereGreaterOrEq(self,column,value,tolerance=0, check_seq=5):
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(0,len(colVec)):
            if colVec[n]>=value-tolerance:
                #print("Index {}: Value {} bigger than {}".format(n,colVec[n],value))
                prev=True
                for i in range(1,check_seq):
                    try:
                        prev= prev and colVec[n+i]>=value-tolerance
                    except IndexError:
                        pass
                    #if prev:
                        #print("And Index {}: Value {} bigger than {}".format(n+i,colVec[n+i],value))
                if prev:
                    return n
        raise IndexError("No Index found where a value is greater than {:7.1E}, invalid limits at column:".format(value)+repr(column)+f"\nColumn is :\n{colVec}")
    
    def getLastIndexWhereSmallerOrEq(self,column,value,tolerance=0, offset=1, check_seq=5):
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(offset,len(colVec)):
            prev=True
            for i in range(1,check_seq):
                prev= prev and colVec[-(n+i)]<=value-tolerance
            if prev:
                return -n
        raise IndexError("No Index found where a value is smaller than {:7.1E}, invalid limits at column:".format(value)+repr(column))
    
    def getLastIndexWhereGreaterOrEq(self,column,value,tolerance=0, check_seq=5):
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(1,len(colVec)):
            prev=True
            for i in range(1,check_seq):
                prev= prev and colVec[-(n+i)]>=value-tolerance
            if prev:
                return -n
        raise IndexError("No Index found where a value is greater than {:7.1E}, invalid limits at column:".format(value)+repr(column))
    
    def getFirstIndexWhereSmallerOrEq(self,column,value,tolerance=0, offset=0, check_seq=5):
        data=self.getData()
        colVec=data[:,column-1]
        for n in range(offset,len(colVec)):
            prev=True
            for i in range(1,check_seq):
                 prev = prev and colVec[n+i]<=value-tolerance
            if prev:
                return n
        raise IndexError("No Index found where a value is greater than {:7.1E}, invalid limits at column:".format(value)+repr(column))
    
    def removeAllButEveryXthLine(self,factor):
        self.setData(self.__data[::factor])
        
    def getDataWithOnlyEveryXthLine(self, factor):
        return Data(self.__data[::factor])
    
    def removeDoubles(self, xCol=0):
        if xCol==0:
            xCol=self.xCol
        tup=np.unique(self.__data[:,xCol-1], return_index=True, axis=0)
        self.setData(self.__data[tup[1]])


