# coding: utf-8

# In[1]:


import matplotlib as mpl

# mpl.use("TkAgg")
# import matplotlib.pyplot
import numpy as np
import pandas as pd
import scipy.interpolate as inter
import copy
import sys
import os
import string
import warnings
import functools
import copy
import inspect
from matplotlib import rc
from Filereader import fileToNpArray
from Data import Data
from Fitter import Fitter
from Plot import Plot
from SpectraPlot import SpectraPlot


# In[2]:


class OLEDPlot(Plot):
    # NaturKonst
    e = 1.6 * 10 ** -19
    K_m = 683  # lm/W
    c = 2.99 * 10 ** 17  # nm/s
    h = 6.63 * 10 ** -34  # J*s
    k_B = 1.38064852 * 10 ** -23  # J/K
    eps_0 = 8.85418781762 * 10 ** -12  # As/(Vm)
    fileFormat = {"separator": "\t", "skiplines": 1}
    # ProgKonst
    chars = list(string.ascii_uppercase)  # alphabetUppercase
    luminosity_file = (
        os.path.dirname(os.path.abspath(inspect.getsourcefile(Data)))
        + "/luminFunction.csv"
    )
    luminosity_fileDataFormat = {
        "separator": ",",
        "skiplines": 0,
        "lastlines": 50,
    }  # custom csv format

    Voltage = 1
    Current = 2
    Current_density = 3
    Luminance = 4
    Radiance = 5
    Current_Efficacy = 6
    Luminous_Efficacy = 7
    EQE = 8

    # defaults
    jvl_file_format_default = {"separator": "\t", "skiplines": 1, "fileEnding": ".uil"}
    spectral_data_format_default = {
        "separator": "\t",
        "skiplines": 40,
        "fileEnding": ".txt",
        "codec": "iso-8859-1",
    }
    pixels_default_qty = 4

    @classmethod
    def ohmicLaw(cls, volt, conductivity):
        return volt * conductivity

    @classmethod
    def shockleyEquation(cls, volt, saturationCurrent, idealityFactor, temperature):
        return saturationCurrent * (
            np.exp(OLEDPlot.e * volt / (cls.k_B * temperature * idealityFactor)) - 1
        )

    @classmethod
    def gurneyMottEquation(cls, volt, thickness, permittivity, mobility):
        return (
            9 * permittivity * cls.eps_0 * mobility * (volt) ** 2 / (8 * thickness ** 3)
        )

    @classmethod
    def oledjVCharFunc(
        cls,
        volt,
        conductivity,
        saturationCurrent,
        idealityFactor,
        temperature,
        thickness,
        permittivity,
        mobility,
        voltOnset,
        sclcOnset,
    ):
        return (
            cls.ohmicLaw(volt, conductivity)
            + np.heaviside(volt - voltOnset, 0)
            * cls.shockleyEquation(
                volt - voltOnset, saturationCurrent, idealityFactor, temperature
            )
            * np.heaviside(sclcOnset - volt, 1)
            + np.heaviside(volt - sclcOnset, 0)
            * cls.gurneyMottEquation(volt, thickness, permittivity, mobility)
        )

    @classmethod
    def candToLumen(cls, cand):
        return cand * 2 * np.pi

    @classmethod
    def calcLumEffic(cls, u, dens, cand):
        return cls.div0(cls.candToLumen(cand), (u * dens))  # lm/W

    @classmethod
    def calcCurEffic(cls, dens, cand):
        return cand / (dens * 10)

    @classmethod
    def doubleLogSlope(cls, volt, dens, V_bi=0):
        oldDict = np.seterr(all="ignore")
        V = np.log(np.abs(volt - V_bi))
        j = np.log(np.abs(dens))
        deriv = np.gradient(j, V)
        np.seterr(**oldDict)
        return deriv

    @classmethod
    def semiLogSlope(cls, volt, dens):
        oldDict = np.seterr(all="ignore")
        j = np.log(dens)
        deriv = np.gradient(j, V)
        np.seterr(**oldDict)
        return deriv

    @classmethod
    def removeZeros(cls, cur_a):
        minVal = 5 * 10 ** -12
        cur_o = [cur if cur != 0.0 else minVal for cur in cur_a]
        return cur_o

    @classmethod
    def generateFileList(
        cls,
        prefix,
        pixels=pixels_default_qty,
        subdir="",
        samples=4,
        fill="_",
        alphaOffset=0,
        truthTable=None,
        postfix="",
        update_by_existing=True,
        century="20",
        fileFormat=jvl_file_format_default,
        files=None,
    ):
        if files is None:
            if update_by_existing:
                files = os.listdir(subdir)
                keys = list(
                    set([(file.split(fill)[0], file.split(fill)[1]) for file in files])
                )
                keys = [key for key in keys if key[0] == prefix]
                fileZ = []
                for key in keys:
                    subfiles = []
                    for file in files:
                        dated_measurement = False
                        try:
                            if (file.split(fill)[2], file.split(fill)[3]) == key:
                                dated_measurement = True
                        except IndexError:
                            dated_measurement = False
                        if (
                            file.split(fill)[0],
                            file.split(fill)[1],
                        ) == key or dated_measurement:
                            subfiles.append(
                                subdir + file[: -len(fileFormat["fileEnding"])]
                            )
                    subfiles.sort()
                    fileZ.append(subfiles)
                fileZ.sort()
                generatedList = fileZ
            else:
                generatedList = [
                    [
                        subdir
                        + prefix
                        + fill
                        + cls.chars[sample + alphaOffset]
                        + postfix
                        + fill
                        + str(pixel + 1)
                        for pixel in range(0, pixels)
                    ]
                    for sample in range(0, samples)
                ]
        else:
            generatedList = files
        if truthTable is None:
            return generatedList
        returningList = []
        for truthTableForSample, sampleList in zip(truthTable, generatedList):
            sampleSubList = []
            for truth, sample in zip(truthTableForSample, sampleList):
                if truth:
                    sampleSubList.append(sample)
            # if sampleList != []:
            returningList.append(sampleSubList)
        return returningList

    @classmethod
    def get_valid_pixel_by_user(
        cls,
        series_indicator,
        jvl_file_format=jvl_file_format_default,
        files=None,
        font_sizes=(8, 10, 12),
        fig_size=(20, 10),
        fill="_",
        **kwargs,
    ):
        valid_pixel = []
        valid_device = []
        SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = font_sizes
        if files is None:
            OLED_fileList = cls.generateFileList(series_indicator, **kwargs)
        else:
            OLED_fileList = files
        # print(OLED_fileList)
        import matplotlib.pyplot as plt

        for sample in OLED_fileList:
            n = 0
            plt.clf()
            plt.figure(figsize=fig_size)
            plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
            plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
            plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
            plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
            plt.rc("figure", titlesize=SMALL_SIZE)  # fontsize of the figure title
            for pixel in sample:
                n += 1
                data = fileToNpArray(pixel, **jvl_file_format)[0]
                plt.plot(
                    data[:, 0],
                    np.absolute(data[:, 1]),
                    label="Current - Px " + pixel[-1],
                    marker=f"${n}$",
                    markersize=BIGGER_SIZE,
                    alpha=0.7,
                )
                plt.plot(
                    data[:, 0],
                    data[:, 2],
                    label="Photocurrent - Px " + pixel[-1],
                    marker=f"${n}$",
                    markersize=BIGGER_SIZE,
                    ls="--",
                    alpha=0.7,
                )
            plt.yscale("log")
            plt.title(sample)
            plt.legend()
            plt.show()
            valid_pixels_input = input("Valid pixels")
            if len(valid_pixels_input) != len(sample):
                raise IndexError("Invalid Input")
            valid_pixel.append([bool(int(pixel)) for pixel in valid_pixels_input])
            valid_device.append(int(valid_pixels_input) != 0)
        return valid_device, valid_pixel

    def __init__(
        self,
        name,
        fileList,
        spectraFile=None,
        fileFormat=jvl_file_format_default,
        title=None,
        pixelsize_mm2=3.8,
        skipSweepBack=True,
        noSweepBackMeasured=False,
        averageSweepBack=False,
        specYCol=2,
        diodYCol=2,
        averageMedian=True,
        specBg=None,  # background column of Jeti spectra
        spectraRange=(390, 790, 401),
        showColAxType=[
            "lin",
            "lin",
            "log",
            "log",
            "log",
            "log",
            "lin",
            "lin",
            "lin",
            "lin",
        ],
        showColAxLim=[None, None, None, None, None, None, None, None, None, None],
        showColLabel=[
            "",
            "Voltage",
            "Current",
            "Current Density",
            "Luminance",
            "Radiance",
            "Current Efficiency",
            "Luminous Efficacy",
            "EQE",
            "Exponent",
        ],
        showColLabel_fileName=[
            "",
            "Voltage",
            "Current",
            "Current Density",
            "Luminance",
            "Radiance",
            "Current Efficiency",
            "Luminous Efficacy",
            "EQE",
            "Exponent",
        ],
        showColLabelUnit=[
            "",
            "Voltage (V)",
            "Current (A)",
            "Current Density (mA/cm²)",
            "Luminance (cd/m²)",
            "Radiance (W/(sr m²))",
            "Current Efficiency (cd/A)",
            "Luminous Efficacy (lm/W)",
            "EQE (%)",
            "Exponent",
        ],
        showColLabelUnitNoTex=[
            "",
            "Voltage (V)",
            "Current (A)",
            "Current Density ($\\tfrac{\\mathsf{mA}}{\\mathsf{cm}^\\mathsf{2}}$)",
            "Luminance ($\\tfrac{\\mathsf{cd}}{\\mathsf{m}^\\mathsf{2}}$)",
            "Radiance ($\\tfrac{\\mathsf{W}}{\\mathsf{sr}\\cdot{}\\mathsf{m}^\\mathsf{2}}$)",
            "Current Efficiency ($\\tfrac{\\mathsf{cd}}{\\mathsf{A}}$)",
            "Luminous Efficacy ($\\tfrac{\\mathsf{lm}}{\\mathsf{W}}$)",
            "EQE (\\%)",
            "Exponent",
        ],
        photodiodeFunctionFile=os.path.dirname(
            os.path.abspath(inspect.getsourcefile(Data))
        )
        + "/luminFunction.csv",
        spectralDataFormat=spectral_data_format_default,  # jeti csv format
        diodeDataFormat={
            "separator": ",",
            "skiplines": 0,
            "lastlines": 50,
        },  # custom csv format
        titleForm="\\textbf{{{} characteristic curve of}}\n\\textbf{{the {} OLED}}",
        legLoc=2,
        samples=None,
        idealDevice=-1,
        maxEqe=5,
        darkCurrent=None,
        darkCurrentValidPoints=(1, 20),
        measErrors={
            "I_p": 5 * 10 ** -12,
            "r": 5 * 10 ** -3,
            "s_max": 0.01608,
            "A_pixel": 10 ** -7,
        },
        V_bi=0,
        curIdeal=False,
        lumThresh=0.1,
        spec_bg_file=None,
        sweepOverride=None,
        invertedDevice=False,
        noVLFilter=False,
        volt_zero_thresh=10 ** -6,
        **kwargs,
    ):
        self.sweepOverride = sweepOverride
        self.averageSweepBack = averageSweepBack
        self.skipSweepBack = skipSweepBack
        self.noSweepBackMeasured = noSweepBackMeasured
        Plot.__init__(
            self,
            name,
            fileList,
            averageMedian=averageMedian,
            showColAxType=showColAxType,
            showColAxLim=showColAxLim,
            showColLabel=showColLabel,
            showColLabelUnit=showColLabelUnit,
            showColLabelUnitNoTex=showColLabelUnitNoTex,
            fileFormat=fileFormat,
            legLoc=legLoc,
            **kwargs,
        )
        # dyn inits
        self.showColLabel_fileName = showColLabel_fileName
        self.pixelsize_mm2 = pixelsize_mm2
        self.diodYCol = diodYCol
        self.spectraRange = spectraRange
        self.spectralDataFormat = spectralDataFormat
        self.diodeDataFormat = diodeDataFormat
        if title is None:
            self.title = titleForm.format(self.showColLabel[self.showCol], name)
        else:
            self.title = title
        if spectraFile is None or spectraFile == "":
            warnings.warn("No SpectraFile given, Radiance and EQE will be wrong!")
        if type(spectraFile) is not list and type(spectraFile) is not tuple:
            self.spectraFiles = (spectraFile,)
        else:
            self.spectraFiles = spectraFile
        self.spec_bg_file = spec_bg_file
        self.specBg = specBg
        self.specYCol = specYCol
        self.photodiodeFunctionFile = photodiodeFunctionFile
        if self.overrideFileList:
            self.samples = samples
        else:
            self.samples = len(self.fileList)
        self.idealDevice = idealDevice
        self.maxEqe = maxEqe
        self.darkCurrent = darkCurrent
        self.darkCurrentValidPoints = darkCurrentValidPoints
        self.measErrors = measErrors
        self.V_bi = V_bi
        self.curIdeal = curIdeal
        self.lumThresh = lumThresh
        self.invertedDevice = invertedDevice
        self.noVLFilter = noVLFilter
        self.volt_zero_thresh = volt_zero_thresh
        # initmethods
        self.exportDataList = copy.deepcopy(self.dataList)
        self.spectralDataList = self.spectraDataImport()[0]
        self.diodeData = self.spectraDataImport()[1]

    def __repr__(self):
        return f"OLEDPlot: {self.name} with files {self.fileList}"

    def spectraDataImport(self):
        bg = self.specBg
        if self.noVLFilter:
            self.luminosityFuncData = Data(
                fileToNpArray(self.luminosity_file, **self.luminosity_fileDataFormat)[0]
            )
            self.luminosityFuncData.processData(Plot.normalize2, yCol=2)
            Plot.equalizeRanges(self.luminosityFuncData)
        diodeFuncData = Data(
            fileToNpArray(self.photodiodeFunctionFile, **self.diodeDataFormat)[0]
        )
        diodeFuncData.processData(Plot.normalize2, yCol=self.diodYCol)
        spectralDataList = []
        lum_interpolator = SpectraPlot.lum_interpolator
        for spectraFileTup in self.spectraFiles:
            if type(spectraFileTup) == tuple or type(spectraFileTup) == list:
                spectraFile = spectraFileTup[0]
                yCol = spectraFileTup[1]
            else:
                spectraFile = spectraFileTup
                yCol = self.specYCol
            try:
                spectralData = Data(
                    fileToNpArray(spectraFile, **self.spectralDataFormat)[0]
                )
                lum_func = lum_interpolator(spectralData.getSplitData2D(yCol=yCol))
                # spectralData.limitData(xLim=[self.spectraRange[0],self.spectraRange[1]])
                xData = np.arange(self.spectraRange[0], self.spectraRange[1])
                yData = lum_func(xData)
            except FileNotFoundError:
                warnings.warn(
                    f"SpectraFile {spectraFile} Not Found, Radiance and EQE will be wrong!"
                )
                spectralData = diodeFuncData
                xData = spectralData.getSplitData2D()[0]
                yData = spectralData.getSplitData2D()[1]
            except TypeError:
                warnings.warn("SpectraFile invalid, Radiance and EQE will be wrong!")
                spectralData = diodeFuncData
                xData = spectralData.getSplitData2D()[0]
                yData = spectralData.getSplitData2D()[1]
            try:
                if self.spec_bg_file is None:
                    bg_func = lum_interpolator(spectralData.getSplitData2D(yCol=bg))
                else:
                    bg_func = lum_interpolator(
                        Data(
                            fileToNpArray(spectraFile, **self.spectralDataFormat)[0]
                        ).getSplitData2D()
                    )
                xData = np.arange(self.spectraRange[0], self.spectraRange[1])
                bgData = bg_func(xData)
                yDataCor = yData - bgData
                spectralData.setData(Data.mergeData((xData, yData, bgData, yDataCor)))
                spectralData.processData(OLEDPlot.absolute, yCol=2)
                spectralData.processData(OLEDPlot.absolute, yCol=4)
                spectralData.processData(OLEDPlot.normalize, yCol=2)
                spectralData.processData(OLEDPlot.normalize, yCol=4)
            except (TypeError):
                spectralData.setData(Data.mergeData((xData, yData)))
                spectralData.processData(OLEDPlot.absolute, yCol=2)
                spectralData.processData(OLEDPlot.normalize, yCol=2)
            Plot.equalizeRanges(spectralData, norm=self.spectraRange)
            spectralDataList.append(spectralData)
        if len(spectralDataList) == 1:
            spectralDataList = spectralDataList * self.samples
        Plot.equalizeRanges(diodeFuncData, norm=self.spectraRange)
        return spectralDataList, diodeFuncData

    def remDarkCurr(self, phot):
        if self.darkCurrent is None:
            a = phot - np.average(
                phot[self.darkCurrentValidPoints[0] : self.darkCurrentValidPoints[1]]
            )
        else:
            a = phot - self.darkCurrent
        return np.maximum(a * 0 + 10 ** -14, np.absolute(a))

    def remZeroCurr_func(self, data):
        volt, curr = data.getSplitData2D(xCol=1, yCol=2)
        for voltage, current in zip(volt, curr):
            if abs(voltage) < self.volt_zero_thresh:

                def remZeroCurr(curr):
                    return curr - current

                return remZeroCurr

        def remZeroCurr(curr):
            return curr

        return remZeroCurr

    def radToCandela(self, rad, spectralData):
        summe = np.sum(
            [
                self.diodeData.getSplitData2D()[1][a]
                * spectralData.getSplitData2D()[1][a]
                for a in range(0, len(self.diodeData.getSplitData2D()[1]))
            ]
        )
        return rad * OLEDPlot.K_m * summe

    def curToDensity(self, cur, pixelsize=None):
        if pixelsize is None:
            return cur * 10 ** 5 / self.pixelsize_mm2
            # converts A to mA/cm²
        return cur * 10 ** 5 / pixelsize
        # converts A to mA/cm²

    def densToCur(self, dens, pixelsize=None):
        if pixelsize is None:
            return dens * self.pixelsize_mm2 / 10 ** 5
            # converts mA/cm² to A
        return dens * pixelsize / 10 ** 5

    def photToCandela(self, phot, pixelsize=None):
        if pixelsize is None:
            return (
                phot * 4.3 * self.pixelsize_mm2 ** (-1) * 10 ** 10
            )  # converts A to cd/m² Correction: pixelsize_mm2
        return (
            phot * 4.3 * pixelsize ** (-1) * 10 ** 10
        )  # converts A to cd/m² Correction: pixelsize_mm2

    def photToRad(self, phot, spectralData, pixelsize=None):
        """According to:

        With PHI_e = Radiant Flux

        I_p=int_0_inf Psi(lambda) dPHI/dlambda dlambda
        I_p=PHI_e* int_0_inf Psi(lambda) dPHI~/dlambda dlambda
        let:
        s=int_0_inf Psi(lambda) dPHI~/dlambda dlambda
        with:
        1 = int_0_inf dPHI~/dlambda dlambda
        Analogously to photToCandela:
        L_e=rad=c*I_p
        c=r**2/(s*A_pixel*A_Photodiode)
        with:
        r=45*10**-3
        A_pixel=10**-6
        A_Photodiode=10**-4

        """
        if not self.noVLFilter:
            raise NotYetImplementedError
        summe = np.sum(
            [
                self.diodeData.getSplitData2D()[1][a]
                * spectralData.getSplitData2D()[1][a]
                for a in range(0, len(self.diodeData.getSplitData2D()[1]))
            ]
        )
        if pixelsize is None:
            return (
                phot * 45 ** 2 * (self.pixelsize_mm2 * summe) ** (-1) * 10 ** 4
            )  # converts A to cd/m² Correction: pixelsize_mm2
        return (
            phot * 45 ** 2 * (pixelsize * summe) ** (-1) * 10 ** 4
        )  # converts A to cd/m² Correction: pixelsize_mm2

    def candToPhotoCurr(self, cand, pixelsize=None):
        if pixelsize is None:
            return cand / (4.3 * self.pixelsize_mm2 ** (-1) * 10 ** 10)
        return cand / (4.3 * pixelsize ** (-1) * 10 ** 10)

    def candToRadiance(self, cand, spectralData):  # converts cd/m² to W/(sr*m²)
        summe = np.sum(
            [
                self.diodeData.getSplitData2D()[1][a]
                * spectralData.getSplitData2D()[1][a]
                for a in range(0, len(self.diodeData.getSplitData2D()[1]))
            ]
        )
        return cand / (OLEDPlot.K_m * summe)

    def radToCandela(self, rad, spectralData):
        if not self.noVLFilter:
            raise NotYetImplementedError
        summe = np.sum(
            [
                self.luminosityFuncData.getSplitData2D()[1][a]
                * spectralData.getSplitData2D()[1][a]
                for a in range(0, len(self.luminosityFuncData.getSplitData2D()[1]))
            ]
        )
        return OLEDPlot.K_m * summe * rad

    def calcEQE(self, dens, rad, spectralData):
        sum2 = np.sum(
            [
                spectralData.getSplitData2D()[1][a]
                / spectralData.getSplitData2D()[0][a]
                for a in range(0, len(spectralData.getSplitData2D()[1]))
            ]
        )
        return (
            (np.pi * rad * OLEDPlot.e)
            / (dens * 10 * OLEDPlot.h * OLEDPlot.c * sum2)
            * 100
        )  # dens*10**4--> mA/m² --> dens*10 --> A/m²

    def theoLimitPhot(self, volt, dens, spectralData):
        sum2 = np.sum(
            [
                spectralData.getSplitData2D()[1][a]
                / spectralData.getSplitData2D()[0][a]
                for a in range(0, len(spectralData.getSplitData2D()[1]))
            ]
        )
        for n in range(0, len(volt)):
            if volt[n] >= 0:
                break
        dens[0:n] = 0
        rad = (self.maxEqe * dens * 10 * OLEDPlot.h * OLEDPlot.c * sum2) / (
            np.pi * OLEDPlot.e * 100
        )
        return self.candToPhotoCurr(self.radToCandela(rad, spectralData))

    def theoLimitCur(self, cand, spectralData):
        sum2 = np.sum(
            [
                spectralData.getSplitData2D()[1][a]
                / spectralData.getSplitData2D()[0][a]
                for a in range(0, len(spectralData.getSplitData2D()[1]))
            ]
        )
        rad = self.candToRadiance(cand, spectralData)
        for n in range(0, len(cand)):
            if cand[n] >= self.lumThresh:
                break
        rad = rad[n:]
        dens = (
            (np.pi * rad * OLEDPlot.e)
            / (self.maxEqe * 10 * OLEDPlot.h * OLEDPlot.c * sum2)
            * 100
        )
        return self.densToCur(dens)

    def addMeasError(self, data):
        # Volterr
        currErr = np.absolute(self.measError["I_p"] + data[1] * 0)
        densErr = np.sqrt(
            (
                self.curToDensity(data[2])
                / self.pixelsize_mm2
                * self.measError["A_pixel"]
            )
            ** 2
            + (self.curToDensity(self.self.measError["I_p"] + data[2] * 0)) ** 2
        )
        lumErr = np.sqrt(
            (self.photToCandela(1) * self.measError["I_p"] + data[3] * 0) ** 2
            + (data[3] * self.measError["r"] / 1) ** 2
        )
        photErr = self.measError["I_p"]
        # wip

    def processFileName(self, option=".pdf"):
        if self.filename is None:
            string = ""
        else:
            string = self.filename
        if self.showCol2 == 0:
            if self.xCol == OLEDPlot.Voltage:
                string += (
                    self.name.replace(" ", "")
                    + self.fill
                    + "OLED_"
                    + self.showColLabel_fileName[self.showCol].replace(" ", "")
                )
            else:
                string += (
                    self.name.replace(" ", "")
                    + self.fill
                    + "OLED_"
                    + self.showColLabel_fileName[self.showCol].replace(" ", "")
                    + "vs"
                    + self.showColLabel_fileName[self.xCol].replace(" ", "")
                )
        else:
            if self.xCol == OLEDPlot.Voltage:
                string += (
                    self.name.replace(" ", "")
                    + self.fill
                    + "OLED_"
                    + self.showColLabel_fileName[self.showCol].replace(" ", "")
                    + "+"
                    + self.showColLabel_fileName[self.showCol2].replace(" ", "")
                )
            else:
                string += (
                    self.name.replace(" ", "")
                    + self.fill
                    + "OLED_"
                    + self.showColLabel_fileName[self.showCol].replace(" ", "")
                    + "+"
                    + self.showColLabel_fileName[self.showCol2].replace(" ", "")
                    + "vs"
                    + self.self.showColLabel_fileName[self.xCol].replace(" ", "")
                )

        if not self.skipSweepBack:
            string += self.fill + "withSweepback"
        if not self.averageMedian:
            string += self.fill + "noMedian"
        if not True in [a[1] for a in self.errors]:
            string += self.fill + "withoutErrors"
        if not self.scaleX == 1:
            string += self.fill + "scaledWith{:03.0f}Pct".format(self.scaleX * 100)
        if self.filenamePrefix is not None or self.filenamePrefix != "":
            self.processFileName_makedirs()
            if self.filenamePrefix is not None:
                if self.filenamePrefix[-1] == os.sep:
                    string = self.filenamePrefix + string
                else:
                    string = self.filenamePrefix + self.fill + string
        return string + option

    def processData_sub_sub(self, deviceData, spectralData, pixelsize=None):
        nList = []
        mList = []
        validData = False
        l = 0
        for data in deviceData:
            data.limitData(xLim=self.xLimOrig, xCol=self.xColOrig)
            if not self.noSweepBackMeasured:
                try:
                    index_sweep=data.getLastIndexWhereGreaterOrEq(self.xColOrig, self.xLimOrig[0])
                    data.setData(data.getData()[:index_sweep])
                except TypeError:
                    pass
            data.processData(self.remDarkCurr, yCol=3)
            #data.processData(self.remZeroCurr_func(data), yCol=2)
            data.processData(OLEDPlot.absolute, yCol=2)
            data.processData(OLEDPlot.removeZeros, yCol=2)

            if self.averageSweepBack and not self.noSweepBackMeasured:
                array = data.getData()
                array1 = array[: int(len(array)) // 2]
                array2 = array[int(len(array)) // 2 :]
                array2 = array2[::-1]
                array = [[a, b] for a, b in zip(array1, array2)]
                result = np.average(array, axis=1)
                data.setData(result)
            subDataList = []
            volt = data.getSplitData2D(xCol=1, yCol=2)[0]
            subDataList.append(volt)
            current = data.getSplitData2D(xCol=1, yCol=2)[1]
            subDataList.append(current)  # Current [2]
            dens = Data.processDataAndReturnArray(
                data, self.curToDensity, pixelsize=pixelsize
            )[:, 1]
            subDataList.append(dens)  # Current_density [3]
            if not self.noVLFilter:
                lum = Data.processDataAndReturnArray(
                    data, self.photToCandela, yCol=3, pixelsize=pixelsize
                )[:, 2]
                rad = self.candToRadiance(lum, spectralData)
            else:
                rad = Data.processDataAndReturnArray(
                    data,
                    self.photToRad,
                    spectralData=spectralData,
                    yCol=3,
                    pixelsize=pixelsize,
                )[:, 2]
                lum = self.radToCandela(rad, spectralData)
            subDataList.append(lum)  # Luminance [4]
            subDataList.append(rad)  # Radiance [5]
            curEffic = OLEDPlot.calcCurEffic(dens, lum)
            subDataList.append(curEffic)  # Current_efficacy
            lumEffic = OLEDPlot.calcLumEffic(subDataList[0], dens, lum)
            subDataList.append(lumEffic)  # Luminous_efficacy
            eqe = self.calcEQE(dens, rad, spectralData)
            subDataList.append(eqe)  # EQE8
            power = self.doubleLogSlope(np.abs(volt - self.V_bi), np.abs(dens))
            subDataList.append(power)  # power9
            data.setData(Data.mergeData(subDataList))
            if self.xLim is not None:
                try:
                    if self.limCol is None:
                        nList.append(
                            data.getFirstIndexWhereGreaterOrEq(self.xCol, self.xLim[0]),
                            check_seq=10,
                        )
                        mList.append(
                            data.getLastIndexWhereSmallerOrEq(self.xCol, self.xLim[1])
                        )
                    else:
                        n = data.getFirstIndexWhereGreaterOrEq(
                            self.limCol, self.xLim[0], check_seq=10
                        )
                        nList.append(n)
                        if self.noSweepBackMeasured or self.skipSweepBack:
                            mList.append(
                                data.getLastIndexWhereSmallerOrEq(
                                    self.limCol, self.xLim[1]
                                )
                            )
                        else:
                            mList.append(-len(data.getData()) // 2)
                    validData = True
                except IndexError as ie:
                    l = len(data.getData()[:, 0])
                    warnings.warn(
                        "Invalid Limits at column "
                        + str(ie)[55]
                        + " with value "
                        + str(ie)[45:52]
                    )
        try:
            if validData:
                n = max(nList)
                m = min(mList)
            else:
                n = l
                m = -1
            for data in deviceData:
                if m == -1:
                    data.setData(data.getData()[n:])
                else:
                    data.setData(data.getData()[n : m + 1])
        except ValueError:
            pass

    def processData_sub(self, dataList, spectralDataList, pixelsizes):
        try:
            if len(pixelsizes) == len(dataList):
                for deviceData, spectralData, pixelsize in zip(
                    dataList, spectralDataList, pixelsizes
                ):
                    self.processData_sub_sub(
                        deviceData, spectralData, pixelsize=pixelsize
                    )
            else:
                raise IndexError("Pixelsize list invalid")
        except TypeError:
            for deviceData, spectralData in zip(dataList, spectralDataList):
                self.processData_sub_sub(deviceData, spectralData, pixelsize=None)

    def processData(self):
        if not self.dataProcessed:
            self.processData_sub(
                self.dataList, self.spectralDataList, self.pixelsize_mm2
            )
        self.dataProcessed = True
        return self.dataList

    def processAllAndExport(self, **kwargs):
        localDataList = self.importData()
        self.processData_sub(localDataList, self.spectralDataList, self.pixelsize_mm2)
        expectData, errData = self.calcCustomAverage(localDataList)
        self.expectData = expectData
        self.deviaData = errData
        return self.exportAllData(expectData=expectData, errData=errData, **kwargs)

    def importData(self):
        if not self.dataImported:
            upper_crop = 0
            if (
                self.skipSweepBack
                and not self.averageSweepBack
                and not self.noSweepBackMeasured
            ):
                if self.sweepOverride is not None:
                    self.dataList = [
                        [
                            Data(
                                fileToNpArray(pixel, **self.fileFormat)[0][
                                    : int(
                                        len(fileToNpArray(pixel, **self.fileFormat)[0])
                                        // 2
                                    )
                                ],
                                xCol=self.xCol,
                                yCol=self.showCol,
                            )
                            if not swOvR
                            else Data(
                                fileToNpArray(pixel, **self.fileFormat)[0],
                                xCol=self.xCol,
                                yCol=self.showCol,
                            )
                            for pixel, swOvR in zip(device, swOv)
                        ]
                        for device, swOv in zip(self.fileList, self.sweepOverride)
                    ]
                else:
                    self.dataList = [
                        [
                            Data(
                                fileToNpArray(pixel, **self.fileFormat)[0][
                                    : int(
                                        len(fileToNpArray(pixel, **self.fileFormat)[0])
                                        // 2
                                    )
                                ],
                                xCol=self.xCol,
                                yCol=self.showCol,
                            )
                            for pixel in device
                        ]
                        for device in self.fileList
                    ]
            else:
                self.dataList = super(OLEDPlot, self).importData()

            # for dataSubList in self.dataList:
            #    for data in dataSubList:
            #        data.removeDoubles(xCol=self.xColOrig)

        return self.dataList

    def afterPlot(self):
        if self.idealDevice >= 0:
            idealDevice = self.idealDevice
            spectralData = self.spectralDataList[idealDevice]
            expVolt = self.expectData[idealDevice].getData()[:, 0]
            expCurr = self.expectData[idealDevice].getData()[:, 1]
            expDens = self.expectData[idealDevice].getData()[:, 2]
            if not self.curIdeal:
                idLum = self.theoLimitPhot(expVolt, expDens, spectralData)
                idPhot = idLum
                data = Data(Data.mergeData((expVolt, expCurr, idPhot)))
            else:
                expLum = self.expectData[idealDevice].getData()[:, 3]
                idCur = self.theoLimitCur(expLum, spectralData)
                expPhot = self.candToPhotoCurr(expLum)
                expPhot = expLum[-len(idCur) :]
                expVolt = expVolt[-len(idCur) :]
                data = Data(Data.mergeData((expVolt, idCur, expPhot)))
            # data.processData(OLEDPlot.remDarkCurr, yCol=3)
            data.limitData(xLim=self.xLimOrig, xCol=self.xColOrig)
            data.limitData(xLim=self.V_bi, xCol=0)
            data.processData(OLEDPlot.absolute, yCol=2)
            subDataList = []
            volt = data.getSplitData2D(xCol=1, yCol=2)[0]
            subDataList.append(volt)
            subDataList.append(data.getSplitData2D(yCol=2)[1])  # Current [2]
            dens = Data.processDataAndReturnArray(data, self.curToDensity)[:, 1]
            subDataList.append(dens)  # Current_density [3]
            lum = Data.processDataAndReturnArray(data, self.photToCandela, yCol=3)[:, 2]
            subDataList.append(lum)  # Luminance [4]
            subDataList.append(self.candToRadiance(lum, spectralData))  # Radiance [5]
            curEffic = OLEDPlot.calcCurEffic(dens, lum)
            subDataList.append(curEffic)  # Current_efficacy
            lumEffic = OLEDPlot.calcLumEffic(subDataList[0], dens, lum)
            subDataList.append(lumEffic)  # Luminous_efficacy
            eqe = self.calcEQE(dens, subDataList[4], spectralData)
            subDataList.append(eqe)  # EQE8
            power = self.doubleLogSlope(np.abs(volt - self.V_bi), np.abs(dens))
            subDataList.append(power)  # power9
            data.setData(Data.mergeData(subDataList))
            self.idealData = data
            try:
                if not self.curIdeal:
                    if self.showCol not in [0, 1, 2, 3, 8, 9]:
                        AX = self.ax.errorbar(
                            *data.getSplitData2D(xCol=self.xCol, yCol=self.showCol),
                            c="#000000",
                            ls=self.ls,
                            label=self.showColLabel[self.showCol]
                            + " at a theo. EQE of "
                            + str(self.maxEqe)
                            + "\\,\\%",
                        )
                    if self.showCol2 not in [0, 1, 2, 3, 8, 9]:
                        AX = self.ax2.errorbar(
                            *data.getSplitData2D(xCol=self.xCol, yCol=self.showCol2),
                            c="#000000",
                            ls=self.ax2ls,
                            label=self.showColLabel[self.showCol2]
                            + " at a theo. EQE of "
                            + str(self.maxEqe)
                            + "\\,\\%",
                        )
                else:
                    if self.showCol not in [0, 1, 4, 5, 6, 7, 8]:
                        AX = self.ax.errorbar(
                            *data.getSplitData2D(xCol=self.xCol, yCol=self.showCol),
                            c="#000000",
                            ls=self.ls,
                            label=self.showColLabel[self.showCol]
                            + " at a theo. EQE of "
                            + str(self.maxEqe)
                            + "\\,\\%",
                        )
                    if self.showCol2 not in [0, 1, 4, 5, 6, 7, 8]:
                        AX = self.ax2.errorbar(
                            *data.getSplitData2D(xCol=self.xCol, yCol=self.showCol2),
                            c="#000000",
                            ls=self.ax2ls,
                            label=self.showColLabel[self.showCol2]
                            + " at a theo. EQE of "
                            + str(self.maxEqe)
                            + "\\,\\%",
                        )
            except:
                raise
        # for a in


class OLEDPlot_db:

    raw_columns = [
        "series",
        "device",
        "stack",
        "emittertype",
        "emitter",
        "jvl_data",
        "jvl_error",
        "spectrum",
        "lt_data",
        "emitter_series",
        "info",
    ]

    @classmethod
    def save_db(cls, db_path, dateframe):
        dataframe.to_pickle(db_path)

    def __init__(self, db_path="oled.db", fill="_"):
        self.fill = fill
        self.db_path = db_path
        try:
            self.__oleds_df = pd.read_pickle(db_path)
            if list(self.__oleds_df.columns) != self.raw_columns:
                raise Exception("Invalid data base type")
        except (FileNotFoundError, ValueError):
            warnings.warn("Error importing db")
            self.__oleds_df = None

    def add_oled(self, oled):
        try:
            self.__oleds_df.loc[
                f"{oled.series}{self.fill}{oled.device}"
            ] = oled.to_dict()
        except AttributeError as e:
            self.__oleds_df = pd.DataFrame(columns=oled.to_dict().keys())
            self.__oleds_df.loc[
                f"{oled.series}{self.fill}{oled.device}"
            ] = oled.to_dict()

    def save_raw_db(self, db_path=None):
        if db_path is None:
            self.__oleds_df.to_pickle(self.db_path)
        else:
            self.__oleds_df.to_pickle(db_path)

    def get_df(self):
        return self.__oleds_df

    def add_max_eqe(self, eqe_thresh=0.1, check_seq=5):
        max_eqes = []
        jvl_series = self.__oled_df[self.raw_columns[5]]
        for data in jvl_series:
            try:
                valid_eqe_start = data.getFirstIndexWhereGreaterOrEq(
                    4, eqe_thresh, check_seq=check_seq
                )
                valid_eqes = data.getData()[:, 7][valid_eqe_start:]
                max_eqe = max(valid_eqes)
                max_eqes.append(max_eqe)
            except IndexError:
                max_eqes.append(0)
        return


class OLED:
    def __init__(
        self,
        series,
        device,
        label,
        stack,
        emittertype,
        emitter,
        jvl_data,
        jvl_error,
        spectrum,
        lt_data=None,
        emitter_series=None,
        info="",
    ):
        self.series = series  # string
        self.device = device  # string
        self.label = label  # string
        self.stack = stack  # stack
        self.emittertype = emittertype  # string
        self.emitter = emitter  # string
        self.jvl_data = jvl_data  # ndarray
        self.jvl_error = jvl_error  # ndarray
        self.spectrum = spectrum  # ndarray
        self.lt_data = lt_data  # ndarray
        self.emitter_series = emitter_series  # string
        self.info = info  # string

    def to_dict(self, series_device=True):
        sel_dict = {}
        if series_device:
            sel_dict.update(dict(series=self.series, device=self.device))
        sel_dict.update(
            dict(
                label=self.label,  # string
                stack=self.stack,  # stack
                emittertype=self.emittertype,  # string
                emitter=self.emitter,  # string
                jvl_data=self.jvl_data,
                jvl_error=self.jvl_error,
                spectrum=self.spectrum,
                lt_data=self.lt_data,
                emitter_series=self.emitter_series,  # string
                info=self.info,  # string
            )
        )
        return sel_dict
