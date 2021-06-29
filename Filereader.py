
# coding: utf-8

# In[1]:

import numpy as np


# In[5]:

def fileToNpArray(filename, 
                  separator=",", 
                  skiplines=1, 
                  backoffset=0, 
                  lastlines=0, 
                  exceptColumns=[], 
                  commaToPoint=False, 
                  lastlineNoNewLine=False, 
                  fileEnding=None, 
                  transpose=False, 
                  ignoreRowCol=None,
                  unitConversionFactors=[1,1],
                  debug=False,
                  codec="utf-8"):
    if fileEnding is None:
        file = open(filename,"r", encoding=codec)
    else:
        file = open(filename+fileEnding,"r",encoding=codec)
    lines = file.readlines()
    if lastlines != 0:
        lines = lines[:-lastlines]
    file.close()
    yAxis = len(lines)
    preArray = []
    descList = []
    fileToNpArray.array = np.asarray([0.0], dtype=np.float64)
    try:
        for x in range(0, skiplines):
            oriItem = lines[x]
            item = oriItem[:-1]
            if separator != " ":
                item = item.replace(" ","")
            line = item.split(separator)
            if separator == " ":
                line = list(filter(None, line))
            descList.append(line)
        for x in range(skiplines, yAxis):
            oriItem = lines[x]
            if x == yAxis-1 and lastlineNoNewLine:
                item=oriItem
            else:
                realBackoffset=-(1+backoffset)
                item = oriItem[:realBackoffset]
            if separator != " ":
                item = item.replace(" ","")
            line = item.split(separator)
            line = list(filter(None, line))
            if commaToPoint==True:
                line=[a.replace(",",".") for a in line]
            preArray.append(line)
        if ignoreRowCol is not None:
            preArray[ignoreRowCol[0]][ignoreRowCol[1]]=0.0
        if (len(exceptColumns)!=0):
            for n in preArray:
                for a,b in zip(exceptColumns, range(0,len(exceptColumns))):
                    del n[a-b]
        if debug:
            print(preArray)
        fileToNpArray.array = np.asarray(preArray, dtype=np.float64)
        if transpose:
            fileToNpArray.array=np.transpose(fileToNpArray.array)
        try:
            for n in range(0,len(fileToNpArray.array[0])):
                fileToNpArray.array[:,n]=fileToNpArray.array[:,n]*unitConversionFactors[n]
        except IndexError:
            pass
        if debug:
            print(fileToNpArray.array)
        #fileToNpArray.array = preArray
    except ValueError as err:
        raise Exception("Error importing file: "+ filename+": "+str(err)+" at line "+str(x+skiplines))
    return fileToNpArray.array, descList

def npArrayToFile(filename, array, preString=None, separator=",", fileEnding=None, skiplines=1, linefeed="\n", **kwargs):
    if fileEnding is None:
        file = open(filename,"w")
    else:
        file = open(filename+fileEnding,"w")
    if preString is not None:
        skiplines-=preString.count(linefeed)
        file.write(preString)
    if skiplines>=1:
        line=separator
        for n in range(0,skiplines):
            file.write(line+linefeed)
    for n in range(0,len(array)):
        line=""
        for element in array[n]:
            line+=str(element)+separator
        line=line[:-len(separator)]
        file.write(line+linefeed)
    file.close()


