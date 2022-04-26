# First we need to import matlab .mat data files uisng scipy
import scipy.io as sio
import numpy as np
import torch
import matplotlib.pyplot as plt

def import_data1(path=None):
    numpy_unpacked=[]
    rawdata = sio.loadmat(path)
    datamatrix=rawdata['datamatrix']
    attributes=datamatrix[:][:11]
    train_data=np.concatenate((datamatrix[0:550],datamatrix[-528:]),axis=0)
    for keys,values in rawdata.items():
        a=1


import_data1("D:\\OneDrive - Indian Institute of Science\\4th Sem\\NSP\\NSP-Grant-Proposal\\MATLAB-COde\\forPython.mat")

