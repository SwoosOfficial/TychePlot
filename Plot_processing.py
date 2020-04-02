#!/usr/bin/env python
# coding: utf-8

import copy
from multiprocessing import Process, Queue, Manager

TIMEOUT_PLOT=30

def initPlot(name, fileList, inputParameters, cls, optionalParameters, xCol=1, showColTup=(2,3), customInputParameters=None):
    local_inputParameters=copy.deepcopy(inputParameters)
    if customInputParameters is not None:
        local_inputParameters.update(customInputParameters)
        try:
            local_optionalParameters=customInputParameters["optionalParameters"]
            local_inputParameters.pop("optionalParameters")
        except KeyError:
            if optionalParameters is not None:
                local_optionalParameters=copy.deepcopy(optionalParameters)
            else:
                local_optionalParameters=None
    else:
        if optionalParameters is not None:
            local_optionalParameters=copy.deepcopy(optionalParameters)
        else:
            local_optionalParameters=None
    if local_optionalParameters is not None and local_optionalParameters["customLims"]:
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
    return plot

def buildPlot(desiredPlot, init_plot_args, queue=None, index=0):
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
    plot_draft=initPlot(*init_plot_args, xCol=desiredPlot["xCol"], showColTup=(desiredPlot["yCol"],yCol2), customInputParameters=cusPara)
    if queue is not None:
        queue.put((index, plot_draft))
        queue.task_done()
        return
    else:
        return plot_draft


def calc_single_plot(plot, queue=None, index=0):
    plot_result=plot.doPlot()
    if queue is not None:
        queue.put((index,plot_result))
        queue.task_done()
        return
    else:
        return plot_result

def calc(desiredPlots, init_plot_args, multithreaded=True):
    #init_plot_args=[name, fileList, inputParameters, cls, optionalParameters]
    if multithreaded:
        building_processes=[]
        plotting_processes=[]
        index=0
        manager=Manager()
        plots_queue=manager.JoinableQueue()
        results_queue=manager.JoinableQueue()
        results=[]
        for desiredPlot in desiredPlots:
            p = Process(target=buildPlot, args=(desiredPlot, init_plot_args, plots_queue, index))
            p.start()
            building_processes.append(p)
            index+=1
        for i in range(0,index):
            plot_draft_tup=plots_queue.get(True,TIMEOUT_PLOT)
            p = Process(target=calc_single_plot(plot_draft_tup[1], results_queue, plot_draft_tup[0]))
            p.start()
            plotting_processes.append(p)
        for p in building_processes:
            p.join()
        for p in plotting_processes:
            p.join()
        for i in range(0,index):
            try:
                results.append(results_queue.get_nowait())
            except:
                pass
        results.sort()
        output=[result[1] for result in results]
    else:
        #singlethreaded
        plotList=[buildPlot(desiredPlot, init_plot_args) for desiredPlot in desiredPlots]
        output=[calc_single_plot(plot_draft) for plot_draft in plotList]
    return output

def export_data(plots, **kwargs):
    for plot in plots:
        plot.processAllAndExport(**kwargs)

def calc_plot_list(plot_prop_list, multithreaded=True, directPlotInput=False):
    if multithreaded:
        results_super=[]
        plotting_super_processes=[]
        manager=Manager()
        results_super_queue=manager.JoinableQueue()
        index=0
        for plot_props in plot_prop_list:
            if directPlotInput:
                p = Process(target=direct_plot, args=(plot_props,results_super_queue, index))
            else:
                p = Process(target=plot, args=(results_super_queue, index), kwargs=plot_props)
            p.start()
            plotting_super_processes.append(p)
            index+=1
        for p in plotting_super_processes:
            p.join()
        for i in range(0,index):
            results_super.append(results_super_queue.get_nowait())
        results_super.sort()
        results_super=[result[1] for result in results_super]
    else:
        if directPlotInput:
            [direct_plot(plot) for plot in plot_prop_list]
        else:
            results_super=[plot(None,0,**plot_props) for plot_props in plot_prop_list]
    return results_super
    

def direct_plot(plot, queue=None, index=0):
    result=plot.doPlot()
    if queue is not None:
        queue.put((index,result))
        queue.task_done()
        return
    else:
        return result 
    
def plot(
         queue,
         index,
         name="",
         fileList=[],
         desiredPlots=[],
         present_params={},
         inputParameters={},
         plot_class=None,
         optionalParameters={},
         feature="both",
         multithreaded=True,
         title="", 
         add_name="",
        ):
    if add_name != "":
        name=name+add_name
    if feature == "both":
        presentPlots=copy.deepcopy(desiredPlots)
        for presentPlot in presentPlots:
            try:
                presentPlot["custom"].update(present_params)
            except KeyError:
                presentPlot["custom"]=present_params      
        desiredPlots=desiredPlots+presentPlots
    elif feature == "default":
        pass
    elif feature == "present":
        for desiredPlot in desiredPlots:
            try:
                desiredPlot["custom"].update(present_params)
            except KeyError:
                desiredPlot["custom"]=present_params   
    else:
        raise ValueError("No Such Feature")
    init_plot_args=[copy.deepcopy(name), copy.deepcopy(fileList), copy.deepcopy(inputParameters), copy.deepcopy(plot_class), copy.deepcopy(optionalParameters)]
    result=copy.copy(calc(desiredPlots, init_plot_args, multithreaded=multithreaded))
    if multithreaded:
        queue.put((index,result))
        queue.task_done()
        return
    else:
        return result 
            
            
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