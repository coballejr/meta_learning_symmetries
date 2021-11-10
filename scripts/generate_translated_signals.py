import numpy as np
from symmetries.toydata import sinc_interp, translate_with_inter

data_dir = '/home/chris/projects/symmetries/data'
out_file = '/translated_signals.npz'

n_examples = 10000
n_samples, n_upsamples = 20, 1000
left_endpoint, right_endpoint = 0, 1
eps = 0.1

u = np.linspace(left_endpoint - eps, right_endpoint + eps, n_upsamples)
s = np.linspace(left_endpoint, right_endpoint, n_samples)

T = s[1] - s[0]
tau = T / 2 # half-pixel translation

x = np.random.rand(n_examples, n_samples)
x_inter = np.zeros((n_examples, n_upsamples))
x_trans = np.zeros((n_examples, n_samples))
s_trans = np.zeros((n_examples, n_samples))

for i, downsamp_sig in enumerate(x):
    sgn = np.sign(np.random.randn())
    y = sinc_interp(downsamp_sig, s, u)
    x_inter[i,:] = y
    shift = sgn*tau
    s_trans[i,:], x_trans[i,:] = translate_with_inter(s, u, y, shift)

np.savez(data_dir + out_file, x = x, x_inter = x_inter, x_trans = x_trans, s_trans = s_trans)
