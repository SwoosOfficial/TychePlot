#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python


inputParameters={}
inputParametersForScaled={}
fileListRefl=[]
fileListTrans=[]
name=""
optionalParameters={
    "customLims":False,
    "xOrigLims":[None,None,None,None],
    "yAxisLims":[None,None,None,None],
    "scaled":False
}
cls=None

def initPlot(xCol=1, showColTup=(2,0), xCol2=0, customInputParameters=None):
    if customInputParameters is not None:
        inputParameters.update(customInputParameters)
        inputParametersForScaled.update(customInputParameters)
    scPlot=None
    plot=ReflectoPlot(name, 
                      fileListRefl, 
                      fileListTrans,
                      xCol=xCol,
                      xCol2=xCol2,
                      showColTup=showColTup,           
                      **inputParameters)
    if scaled:
        scPlot=ReflectoPlot(name,
                            fileListRefl, 
                            fileListTrans,
                            xCol=xCol,
                            xCol2=xCol2,
                            showColTup=showColTup,
                            **inputParametersForScaled)
    if scPlot is None:
        return [plot]
    return [plot,scPlot]

def buildPlotList(desiredPlot):
    try:
        yCol2=desiredPlot["yCol2"]
    except KeyError:
        yCol2=0
    try:
        cusPara=desiredPlot["custom"]
    except KeyError:
        cusPara=None
    if desiredPlot["yCol"]==0 or desiredPlot["xCol"]==0:
        raise
    return initPlot(xCol=desiredPlot["xCol"], showColTup=(desiredPlot["yCol"],yCol2), customInputParameters=cusPara)


def processPlotPair(plotpair):
    return [plot.doPlot() for plot in plotpair]


def calc(name_local, fileListRefl_local, fileListTrans_local, desiredPlots, inputParameters_local, cls_local, inputParametersForScaled_local, optionalParameterDict=None, multithreaded=True):
    global name
    global fileListRefl
    global fileListTrans
    global cls
    global inputParameters
    global inputParametersForScaled
    global optionalParameters
    name=name_local
    fileListRefl=fileListRefl_local
    fileListTrans=fileListTrans_local
    inputParameters=inputParameters_local
    inputParametersForScaled=inputParametersForScaled_local
    if optionalParameterDict is not None:
        optionalParameters=optionalParameterDict
    if multithreaded:
        import os
        from multiprocessing import Pool
        pool = Pool(os.cpu_count())
        plotList=pool.map(buildPlotList,desiredPlots)
        multiOutput=pool.map(processPlotPair,plotList)
        pool.close()
    else:
        #singlethreaded
        plotList=[buildPlotList(desiredPlot) for desiredPlot in desiredPlots]
        multiOutput=[processPlotPair(plotPair) for plotPair in plotList]
    plots=[[plotOutput[0] for plotOutput in plotPair] for plotPair in multiOutput]
    files=[[plotOutput[1] for plotOutput in plotPair] for plotPair in multiOutput]
    return (plots,files)

def export_data(plots):
    for plotpair in plots:
        for plot in plotpair:
            plot.processAllAndExport()

