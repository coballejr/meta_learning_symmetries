from symmetries.toydata import Cn
import numpy as np

# set seed for reproducibility
np.random.seed(42)

# define directory and output file
data_dir = '/home/chris/projects/symmetries/data'
out_name = '/c6_signals.npz'

# generate signals
num_signals = 10000
n = 6
grp = Cn(6)
k = np.zeros(n)
k[0:3] = np.random.randn(3)

H = np.zeros((num_signals, n))
F = np.zeros((num_signals, n))
for i in range(num_signals):
    f = np.random.randn(n)
    h = grp.convolve(f, k)
    F[i,:] = f
    H[i,:] = h
    
np.savez(data_dir + out_name, H = H, k = k, F = F)

