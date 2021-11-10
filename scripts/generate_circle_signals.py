# imports
from symmetries.toydata import Filter
import numpy as np

# set seed for reproducibility
np.random.seed(42)

# define directory and output file
data_dir = '/home/chris/projects/symmetries/data'
out_name = '/circle_signals.npz'

num_tasks, num_examples, signal_length, size = 500, 20, 70, 3
y = np.zeros((num_tasks, num_examples ,signal_length))
x = np.random.randn(num_tasks, num_examples ,signal_length)
w = np.zeros((num_tasks, size))
for t in range(num_tasks):
    values = np.random.randn(size)
    w[t, :] = values
    filt = Filter(size, values)
    for i, f in enumerate(x[t]):    
        y[t, i, :] = filt.convolve(f)
np.savez(data_dir + out_name, x = x, y = y, w = w )

