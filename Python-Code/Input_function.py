# First we need to import matlab .mat data files uisng scipy
import scipy.io as sio
import numpy as np
import torch
import matplotlib.pyplot as plt

def import_data1(path=None):
    rawdata = sio.loadmat(path)


import_data1("D:\\OneDrive - Indian Institute of Science\\4th Sem\\NSP\\NSP-Grant-Proposal\\MATLAB-COde\\tempvital.mat")

