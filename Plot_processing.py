#!/usr/bin/env python
# coding: utf-8

import copy

inputParameters={}
inputParametersForScaled={}
fileList=[]
name=""
optionalParameters={
    "customLims":False,
    "xOrigLims":[None,None,None,None],
    "yAxisLims":[None,None,None,None],
    "scaled":False
}
cls=None

def initPlot(xCol=1, showColTup=(2,3), customInputParameters=None):
    local_inputParameters=copy.deepcopy(inputParameters)
    local_inputParametersForScaled=copy.deepcopy(inputParametersForScaled)
    if customInputParameters is not None:
        local_inputParameters.update(customInputParameters)
        local_inputParametersForScaled.update(customInputParameters)
        try:
            local_optionalParameters=customInputParameters["optionalParameters"]
            local_inputParameters.pop("optionalParameters")
            local_inputParametersForScaled.pop("optionalParameters")
        except KeyError:
            local_optionalParameters=copy.deepcopy(optionalParameters)
    else:
        local_optionalParameters=copy.deepcopy(optionalParameters)
    scPlot=None
    if local_optionalParameters["customLims"]:
        plot=cls(
                    name,
                    fileList,
                    xCol=xCol,
                    showColTup=showColTup,
                    xLimOrig=local_optionalParameters["xOrigLims"][showColTup[0]],
                    showColAxLim=local_optionalParameters["yAxisLims"], 
                    **local_inputParameters
        )
        if local_optionalParameters["scaled"]:
            scPlot=cls(
                            name,
                            fileList,
                            xCol=xCol,
                            showColTup=showColTup,
                            xLimOrig=local_optionalParameters["xOrigLims"][showColTup[0]],
                            showColAxLim=local_optionalParameters["yAxisLims"], 
                            **local_inputParametersForScaled
            )
    else:
        plot=cls(
                    name,
                    fileList,
                    xCol=xCol,
                    showColTup=showColTup,
                    **local_inputParameters
        )
        if local_optionalParameters["scaled"]:
            scPlot=cls(
                            name,
                            fileList,
                            xCol=xCol,
                            showColTup=showColTup,
                            **local_inputParametersForScaled
            )
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


def calc(name_local, fileList_local, desiredPlots, inputParameters_local, cls_local, inputParametersForScaled_local, optionalParameterDict=None, multithreaded=True):
    global name
    global fileList
    global cls
    global inputParameters
    global inputParametersForScaled
    global optionalParameters
    name=name_local
    fileList=fileList_local
    cls=cls_local
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

