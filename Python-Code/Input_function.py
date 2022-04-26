import scipy.io as sio
import numpy as np


def import_data1(path=None):

    rawdata = sio.loadmat(path)
    datamatrix = rawdata['datamatrix'].astype(np.float32)

    for i in range(datamatrix.shape[1]):
        datamatrix[:, i] = nan_helper(datamatrix[:, i])
   # train_data = np.concatenate((datamatrix[0:550], datamatrix[-528:]), axis=0)
    train_data = datamatrix[:-200]
    test_data = datamatrix[-200:]

    train_attributes = train_data[:, :11]
    train_labels = train_data[:, 11:]/100

    test_attributes = test_data[:, :11]
    test_labels = test_data[:, 11:]/100

    return [train_attributes, train_labels, test_attributes, test_labels]


def nan_helper(y):
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


import_data1("D:\\OneDrive - Indian Institute of Science\\4th Sem\\NSP\\NSP-Grant-Proposal\\MATLAB-COde\\forPython.mat")
