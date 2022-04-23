from Input_function import import_data1
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
[data,time]=import_data1(None)

from pyhht.visualization import plot_imfs
from pyhht import EMD
decomposer = EMD(data[0])
imfs = decomposer.decompose()
plot_imfs(data[0], imfs, time[0]);
x=sp.hilbert(imfs[0])
for imf in imfs[1:]:
    x+=sp.hilbert(imf)

plt.plot(x)
plt.show()