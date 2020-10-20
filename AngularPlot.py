
# coding: utf-8

# In[ ]:


import numpy as np
import collections
import itertools
import os

from Data import Data
from Plot import Plot

cur_dir=os.path.dirname(os.path.realpath(__file__))
subdir="angMeas"
BLZ_path_dict = {
            100: os.path.join(cur_dir,subdir,"BLZ_100_780nm.wl"), 
            150: os.path.join(cur_dir,subdir,"BLZ_150_500nm.wl"), 
            300: os.path.join(cur_dir,subdir,"BLZ_300_500nm.wl"), 
}

class Line:
    def __init__(self, data):
        self.pos = [float(val) for val in data[1:5]]
        self.photocur_date = data[5]
        self.photocur = data[6]
        self.spec_data = data[7]
        self.spec_empty = not (len(data)>9)
        if(self.spec_empty):
            self.spec = []           
            #print(f"Emtpy Spec {len(data)}")
        else:
            self.spec = [float(val) for val in data[8:-1]]
    def getPos(self):
        return self.pos
    def getTheta(self):
        return self.pos[1]
    def getSize(self):
        #print(len(self.spec))
        return len(self.spec)
    def getOwisPol(self):
        return self.pos[3]
    def getSpec(self):
        return np.array(self.spec)
    def InterpolateSpec(self, line1, line2):
        self.spec = [np.interp(self.getTheta(), [line1.getTheta(), line2.getTheta()], [val1, val2]) for val1, val2 in zip(line1.getSpec(), line2.getSpec())]

class Measurement:
    def __init__(self, data, key, wavelengths):
        pass
    @staticmethod
    def getMeasType(data, key, wavelengths):
        if key == 'ANGULAR_MEAS' or key =='ANGULARMEAS0_MEAS' or key == 'ANGULAR0':
            return AngularMeasurement(data, key, wavelengths)
        else:
            return None
    
class AngularMeasurement(Measurement):
    def __init__(self, data, key, wavelengths):
        self.w = wavelengths
        #see if we have more than one measurement
        measurements = []
        zero_dat = data[0].getOwisPol()
        pos = 0
        for i, val in enumerate(data):
            if(val.getOwisPol()==zero_dat and data[i-1].getOwisPol()!=zero_dat) and i>0:
                measurements.append(data[pos:i-1])
                pos = i
                #print(pos)
                #print(f"{val.getOwisPol()} {val.spec_data} {data[i-1].getOwisPol()} {data[i-1].spec_data}")
                
                #del data[0:i-1]
        if(pos!=(len(data)-1)):
            measurements.append(data[pos:-1])
            #now we have all data
        #print(len(measurements))
        #print(f"Num Angular Measurements: {len(measurements)}")
        self.angles = []
        self.dat = []
        self.pol_pos = []
        for meas in measurements:
            #print(len(meas))
            #read the angles and make sure we get each angle only once
            _angles = [val.getPos()[1] for val in meas]
            self.angles.append([])
            for val in _angles:
                if not val in self.angles[-1]:
                    self.angles[-1].append(val)
            #read the pol-positions and make sure we get each only once
            _pol_pos = [val.getPos()[3] for val in meas]
            self.pol_pos.append([])
            for val in _pol_pos:
                if not val in self.pol_pos[-1]:
                    self.pol_pos[-1].append(val)
            self.dat.append(np.ndarray((len(self.pol_pos[-1]), len(self.angles[-1]), meas[0].getSize()), dtype = float))
            for line in meas:
                idx_a = self.angles[-1].index(line.getTheta())
                idx_p = self.pol_pos[-1].index(line.getOwisPol())
                self.dat[-1][idx_p][idx_a] = line.getSpec()
    def getIndex(self, val, _list):
        #print(_list)
        return np.argmin(np.abs(np.array(_list)-val))
    def getPeakWavelength(self, index = 0, w1 = 400, w2 = 800):
        dat = self.getDataAtAngle(self.getMinimumAngle(), 0, index)
        idx_min = self.getIndex(w1, self.w)
        _slice = self.getSlice(w1, w2)
        #rint(f"{np.argmax(dat[idx_min:idx_max])} ")
        return self.w[np.argmax(dat[_slice])+_slice.start]
    def getDataAtWavelength(self, w, p, index = 0):
        w_index = self.getIndex(w, self.w)
        return [val[w_index] for val in self.dat[index][self.getIndex(p, self.pol_pos)]]
    def getDataAtAngle(self, p, a = None, index = 0):
        if a is None:
            a = self.getMinimumAngle(index = index)
        return list(self.dat[index][self.getIndex(p, self.pol_pos)][self.getIndex(a, self.angles)])
    def getDataAt(self, w, a, p, index):
        return self.dat[index][self.getIndex(p, self.pol_pos)][self.getIndex(a, self.angles)][self.getIndex(w, self.w)]
    def getAngles(self, p, index = 0):
        return self.angles[index]
    def getPol(self, index = 0):
        return self.pol_pos[index]
    def getMinimumAngle(self, index =0):
        return np.min(np.abs(self.angles[index]))
    def getWavelengths(self):
        return self.w
    def getNumMeasurements(self):
        return len(self.dat)
    def getSlice(self, w1 = 420, w2 = 1050):
        return slice(self.getIndex(w1, self.w),self.getIndex(w2,self.w))

class AngularMeasFile:
    def __init__(self, filename, wavelength, BLZFiles = BLZ_path_dict):
        self.w_data = [float(line) for line in open(BLZFiles[wavelength])]
        file = open(filename)
        _data = collections.defaultdict(list)
        for line in file:
            _key = line.split("\t")[0]
            _data[_key].append( Line(line.split("\t")))
            if len(_data[_key])>3 and (_data[_key][-2].spec_empty):
                _data[_key][-2].InterpolateSpec(_data[_key][-3], _data[_key][-1])            
        #read the file and all the data
        self.data={key: Measurement.getMeasType(val,key,self.w_data ) for key, val in _data.items()}
    def getMeasurementTypes(self):
        return self.data.keys()
    def getMeasCounts(self):
        return {key: len(val) for key, val in self.data.items()}
    def getMeas(self, key):
        return self.data[key]

class AngularPlot(Plot):
    
    @classmethod
    def cosine(cls, x, amp):
        return amp*np.cos(x/360*2*np.pi)
    
    def __init__(self,
                 name,
                 fileList,
                 BLZFiles = BLZ_path_dict,
                 BLZFile= 150,
                 index=0,
                 meas_type='ANGULAR_MEAS',
                 wavelengths=-1,
                 angles=None,
                 showColAxType=["lin","lin","lin","lin","lin"],
                 showColAxLim=[None,None,None,None,None],
                 showColLabel= ["", "Angle", "Intensity (s-polarised)", "Intensity (p-polarised)", "Intensity (s-polarised)", "Intensity (p-polarised)"],
                 showColLabelUnit=["", "Angle (°)", "s-polarised Intensity (a.u.)", "p-polarised Intensity (a.u)", "s-polarised Intensity (a.u.)", "p-polarised Intensity (a.u)"],
                **kwargs):
        self.BLZFiles=BLZFiles
        self.BLZFile=BLZFile
        self.index=index
        self.meas_type=meas_type
        self.wavelengths=wavelengths
        self.angles=angles
        self.plotSpectrum=plotSpectrum
        Plot.__init__(self, name, fileList, dataImported=True, showColAxType=showColAxType,
                 showColAxLim=showColAxLim,
                 showColLabel=showColLabel,
                 showColLabelUnit=showColLabelUnit, **kwargs)
        self.dataList=self.importData()
        
    def importData(self):
        dataList=[]
        index=self.index
        wavelengths=
        angles=
        for sampleList in self.fileList:
            dataSubList=[]
            for fileZ in sampleList:
                meas = AngularMeasFile(fileZ, self.BLZFile, BLZFiles=self.BLZFiles)
                a_meas = meas.getMeas(self.meas_type)
                if(self.wavelength == -1):
                    w = a_meas.getPeakWavelength(index)
                else:
                    w = self.wavelength
                if (self.angle == -1):
                    angle = None    
                else:
                    angle = self.angle
                if self.plotSpectrum:
                    data=Data(Data.mergeData([a_meas.getWavelengths(),a_meas.getDataAtAngle(0, angle, index), a_meas.getDataAtAngle(90, angle, index) ]))
                    self.showColLabel[1]="Wavelength"
                    self.showColLabelUnit[1]="Wavelength (nm)"
                else:
                    data=Data(Data.mergeData([a_meas.getAngles(0, index), a_meas.getDataAtWavelength(w, 0, index),a_meas.getDataAtWavelength(w, 90, index)]))
                data.limitData(xLim=self.xLimOrig)
                dataSubList.append(data)
            dataList.append(dataSubList)
        return dataList
        
