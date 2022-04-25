# First we need to import matlab .mat data files uisng scipy
import scipy.io as sio
import numpy as np
import torch
import matplotlib.pyplot as plt
#rawdata = sio.loadmat(path)
def import_data1(path=None):
    if path is None:
        time=np.linspace(0,20,1000)
        signal=np.sin(time)+0.5*np.sin(time/2)+0.25*np.cos(time*2)+0.1*np.random.rand(1000)
        return [np.array([signal]*6), np.array([time]*6)]
    else:
        data = sio.loadmat(path)
        return data

