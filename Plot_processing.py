#!/usr/bin/env python
# coding: utf-8

import copy

inputParameters={}
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
    if customInputParameters is not None:
        local_inputParameters.update(customInputParameters)
        try:
            local_optionalParameters=customInputParameters["optionalParameters"]
            local_inputParameters.pop("optionalParameters")
        except KeyError:
            local_optionalParameters=copy.deepcopy(optionalParameters)
    else:
        local_optionalParameters=copy.deepcopy(optionalParameters)
    if local_optionalParameters["customLims"]:
        try:
            xLimOrig=local_optionalParameters["xOrigLims"][showColTup[0]]
        except TypeError:
            xLimOrig=local_optionalParameters["xOrigLims"][showColTup[0][0]]
        plot=cls(
                    name,
                    fileList,
                    xCol=xCol,
                    showColTup=showColTup,
                    xLimOrig=xLimOrig,
                    showColAxLim=local_optionalParameters["yAxisLims"], 
                    **local_inputParameters
        )

    else:
        plot=cls(
                    name,
                    fileList,
                    xCol=xCol,
                    showColTup=showColTup,
                    **local_inputParameters
        )
    return [plot]

def buildPlotList(desiredPlot):
    try:
        yCol2=desiredPlot["yCol2"]
    except KeyError:
        yCol2=0
    try:
        xCol2=desiredPlot["xCol2"]
    except KeyError:
        xCol2=0
    try:
        cusPara=desiredPlot["custom"]
        if cusPara is None:
            cusPara={}
    except KeyError:
        cusPara={}
    cusPara["xCol2"]=xCol2
    if desiredPlot["yCol"]==0 or desiredPlot["xCol"]==0:
        raise
    return initPlot(xCol=desiredPlot["xCol"], showColTup=(desiredPlot["yCol"],yCol2), customInputParameters=cusPara)


def processPlotPair(plotpair):
    return [plot.doPlot() for plot in plotpair]


def calc_plotList(plotList, pool=None):
    if pool is None:
        return [processPlotPair(plotPair) for plotPair in plotList]
    return pool.map(processPlotPair,plotList)


def calc(name_local, fileList_local, desiredPlots, inputParameters_local, cls_local, optionalParameterDict=None, multithreaded=True):
    global name
    global fileList
    global cls
    global inputParameters
    global optionalParameters
    name=name_local
    fileList=fileList_local
    cls=cls_local
    inputParameters=inputParameters_local
    if optionalParameterDict is not None:
        optionalParameters=optionalParameterDict
    if multithreaded:
        import os
        from multiprocessing import Pool
        pool = Pool(os.cpu_count())
        plotList=pool.map(buildPlotList,desiredPlots)
        multiOutput=calc_plotList(plotList, pool=pool)
        pool.close()
    else:
        #singlethreaded
        plotList=[buildPlotList(desiredPlot) for desiredPlot in desiredPlots]
        multiOutput=[processPlotPair(plotPair) for plotPair in plotList]
    plots=[[plotOutput[0] for plotOutput in plotPair] for plotPair in multiOutput]
    files=[[plotOutput[1] for plotOutput in plotPair] for plotPair in multiOutput]
    return (plots,files)


def export_data(plots, **kwargs):
    for plotpair in plots:
        for plot in plotpair:
            plot.processAllAndExport(**kwargs)

def plot(
         name,
         fileList,
         desiredPlots,
         present,
         inputParameters,
         plot_class,
         optionalParameters={},
         multithreaded=True,
         title="", 
         add_name="",
         feature="both",
        ):
    if add_name != "":
        name=name+add_name
    if feature == "both":
        presentPlots=copy.deepcopy(desiredPlots)
        desiredPlots=desiredPlots+[presentPlot["custom"].update(present) for presentPlot in presentPlots]
    elif feature == "default":
        pass
    elif feature == "present":
        desiredPlots=[presentPlot["custom"].update(present) for presentPlot in desiredPlots]
    else:
        raise ValueError("No Such Feature")
    return calc(name, fileList, desiredPlots, inputParameters, plot_class, optionalParameterDict=optionalParameters, multithreaded=multithreaded)
            
            
def add_to_plot(plot, func, filename=None, labels=None, ls=None, errors=[False,False], show=[True, True], showLines=[True, True], showMarkers=[False,False]):
    expect_list,devia_list = func(plot)
    if labels is not None:
        for label in labels:
            plot.labels.append(label)
    else:
        plot.labels.append("Unknown")
    if filename is not None:
        plot.filename=filename
    for expect, devia in zip(expect_list, devia_list):
        plot.expectData.append(expect)
        plot.deviaData.append(devia)
        try:
            plot.fitterList.append(None)
        except:
            pass
        plot.errors.append(errors)
        plot.show.append(show)
        plot.showLines.append(showLines)
        plot.showMarkers.append(showMarkers)
    if ls is not None:
        plot.ls=ls
    plot.doPlot()
    return plot