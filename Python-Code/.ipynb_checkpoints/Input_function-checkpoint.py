# First we need to import matlab .mat data files uisng scipy
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def import_data(path=None):
    if path is None:
        return np.random.rand(6, 1000)
    else:
        data = sio.loadmat(path)
        return data

