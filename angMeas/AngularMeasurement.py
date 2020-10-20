import numpy as np
import collections
import itertools
import os

cur_dir=os.path.dirname(os.path.realpath(__file__))

BLZFiles = {
            100: os.path.join(cur_dir,"BLZ_100_780nm.wl"), 
            150: os.path.join(cur_dir,"BLZ_150_500nm.wl"), 
            300: os.path.join(cur_dir,"BLZ_300_500nm.wl"), 
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
        if key == 'ANGULAR_MEAS' or key =='ANGULARMEAS0_MEAS':
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
    def getDataAtAngle(self, a, p, index = 0):
        return self.dat[index][self.getIndex(p, self.pol_pos)][self.getIndex(a, self.angles)]
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
    def __init__(self, filename, wavelength):
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

import matplotlib.pyplot as plt
def PlotMeasurement(filename,wavelength = -1, BLZFile= 150, index = 0, savepath = ""):
    file = filename
    meas = AngularMeasFile(file, BLZFile)
    a_meas = meas.getMeas('ANGULAR_MEAS')
    if savepath:
        file_begin, file_ending = savepath.split(".")
    sample = filename.split("/")[-1].split(".")[0]
    for index in range(a_meas.getNumMeasurements()):
        if(wavelength == -1):
            w = a_meas.getPeakWavelength(index)
        else:
            w = wavelength
        plt.clf()
        plt.title(f"Angular Spectrum of {sample} Measurement #index; Data at {w}nm")
        plt.plot( a_meas.getAngles(90, index),a_meas.getDataAtWavelength(w, 90, index), label = f"ppol")
        plt.plot( a_meas.getAngles(90, index),a_meas.getDataAtWavelength(w, 0, index), label = f"spol")
        plt.legend()
        plt.ylim(bottom = 0)
        plt.xlim(np.amin(a_meas.getAngles(90, index)), np.amax(a_meas.getAngles(90,index)))
        plt.xlabel(f"Angle (°)")
        plt.ylabel(f"Intensity (a.u.)")
        if savepath:
            print(file_begin)
            print(file_ending)
            plt.savefig(file_begin+f"_{index}"+file_ending)
        plt.show()
def PlotSpectrum(filename,angle = -1,w1 = 400, w2 = 800, BLZFile = 150, index = 0, savepath = ""):
    file = filename
    meas = AngularMeasFile(file, BLZFile)
    a_meas = meas.getMeas('ANGULAR_MEAS')
    _slice = a_meas.getSlice(w1, w2)
    file_ending = savepath.split(".")[-1]
    file_begin = "".join(savepath.split(".")[0:-1])
    sample = filename.split("/")[-1].split(".")[0]
    for index in range(a_meas.getNumMeasurements()):
        if(angle == -1):
            a = a_meas.getMinimumAngle(index)
        else:
            a = angle
        w = a_meas.getWavelengths()[_slice]
        plt.clf()
        plt.title(f" Spectrum of {sample} Measurement #{index}; Data at {a} degrees")
        plt.plot( w,a_meas.getDataAtAngle(a, 90, index)[_slice], label = f"ppol")
        plt.plot( w,a_meas.getDataAtAngle(a, 0, index)[_slice], label = f"spol")
        plt.ylim(ymin = 0)
        plt.xlim(w1, w2)
        plt.xlabel(f"Wavelength (nm)")
        plt.ylabel(f"Intensity (a.u.)")
        plt.legend()
        if savepath:
            plt.savefig(file_begin+f"_{index}"+file_ending)
        plt.show()
def CompareMeasurements(filenames, wavelength = -1, BLZFile = 150, index = 0, savepath = ""):

    measfiles = [AngularMeasFile(file, BLZFile) for file in filenames]
    a_meas = [meas.getMeas('ANGULAR_MEAS') for meas in measfiles]
    if wavelength == -1:
        w = a_meas[0].getPeakWavelength(index)
    else: 
        w= wavelength
    min_angles = np.array([meas.getMinimumAngle(index) for meas in a_meas])
    min_angles = np.array([20 for meas in a_meas])
    min_angle = np.amax(min_angles)
    values_s = [meas.getDataAt(w, min_angle, 0, index) for meas in a_meas]
    values_p = [meas.getDataAt(w, min_angle, 90, index) for meas in a_meas]
    corr_s = [values_s[0]/val for val in values_s]
    corr_p = [values_p[0]/val for val in values_p]
    plt.clf()
    names = [s.split("/")[-1].split(".")[0] for s in filenames]
    plt.title(f"Comparing Measurements: {' / '.join(names)} ppol")
    for i,meas in enumerate(a_meas):
        plt.plot( meas.getAngles(90, index),np.array(meas.getDataAtWavelength(w, 90, index))*corr_p[i], label = f"Measurement #{i}")
    plt.legend()
    plt.xlim(np.amin(meas.getAngles(90, index)), np.amax(meas.getAngles(90,index)))
    plt.ylim(bottom = 0)
    plt.xlabel("Angle (°)")
    plt.ylabel("Scaled Intensity (a.u.)")
    if savepath:
        plt.savefig(savepath)
    plt.show()
    plt.clf()
    plt.title(f"Comparing Measurements: {' / '.join(names)} spol")
    for i,meas in enumerate(a_meas):
        plt.plot( meas.getAngles(0, index),np.array(meas.getDataAtWavelength(w, 0, index))*corr_s[i], label = f"Measurement #{i}")
    plt.legend()
    plt.ylim(bottom = 0)
    plt.xlim(np.amin(meas.getAngles(90, index)), np.amax(meas.getAngles(90,index)))
    plt.xlabel("Angle (°)")
    plt.ylabel("Scaled Intensity (a.u.)")
    if savepath:
        plt.savefig(savepath)
    plt.show()
