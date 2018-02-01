
# coding: utf-8

# In[1]:

import numpy as np


# In[5]:

def fileToNpArray(filename, separator=",", skiplines=1, backoffset=0, lastlines=0, exceptColumns=[], commaToPoint=False, lastlineNoNewLine=False):
    file = open(filename,"r")
    lines = file.readlines()
    if lastlines != 0:
        lines = lines[:-lastlines]
    file.close()
    yAxis = len(lines)
    preArray = []
    descList = []
    fileToNpArray.array = []
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
            #if separator == " ":
            line = list(filter(None, line))
            if commaToPoint==True:
                line=[a.replace(",",".") for a in line]
            preArray.append(line)
        if (len(exceptColumns)!=0):
            for n in preArray:
                for a,b in zip(exceptColumns, range(0,len(exceptColumns))):
                    del n[a-b]
        fileToNpArray.array = np.asarray(preArray, dtype=np.float64)
        #fileToNpArray.array = preArray
    except ValueError as err:
        raise Exception("Error importing file: "+ filename+": "+str(err)+" at line "+str(x+skiplines))
        """skiplinesBoolean = input("Is this the first line of data: " + lines[skiplines] + "\n y/n?")
        if skiplinesBoolean == "y" or skiplinesBoolean == "Y":
            separatorBoolean = input("Is this the separator of data, coulums: \"" + separator + "\"\n y/n?")
            if separatorBoolean == "y" or skiplinesBoolean == "Y":
                raise ValueError("There was an error creating the array from the Data.")
            else:
                separator = input("Please specify the new separator:")
        else:
            skiplinesOperator = input("How many lines are description/nondata? \nType \"next\" for trying the next line.")
            if skiplinesOperator.isdigit():
                skiplines = int(skiplinesOperator)
            else:
                skiplines += 1 
        fileToNpArray.array, descList = fileToNpArray(filename, separator=separator, skiplines=skiplines)"""
    return fileToNpArray.array, descList




