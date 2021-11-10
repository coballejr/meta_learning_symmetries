# imports
from symmetries.training_utils import SyntheticLoader, train, test, InnerOptBuilder
from symmetries.modules import ShareLinearFull
import numpy as np
import torch
import matplotlib.pyplot as plt

# load data
data_dir = '/home/chris/projects/symmetries/data/'
data_file = 'circle_signals.npz'
data = np.load(data_dir + data_file)
loader = SyntheticLoader(data)

# build model
in_features, out_features, bias, latent_size = 70, 70, False, 3
mod = ShareLinearFull(in_features, out_features, bias, latent_size)
mod = mod.double()
# construct optimizers
inner_opt = 'maml'
init_inner_lr = 1e-1
lr_mode = 'per_layer'
outer_lr = 1e-3

inner_opt_builder = InnerOptBuilder(mod, inner_opt, init_inner_lr, "learned", lr_mode)
meta_opt = torch.optim.Adam(inner_opt_builder.metaparams.values(), lr = outer_lr)

# train
training_losses, test_losses = [], []
num_inner_steps = 1
num_outer_steps = 1000

for step_idx in range(num_outer_steps):
        data, _filters = loader.next(32, "train")
        train_loss = train(step_idx, data, mod, inner_opt_builder, meta_opt, num_inner_steps)
        training_losses.append(train_loss)
        if step_idx == 0 or (step_idx + 1) % 1 == 0:
            test_data, _filters  = loader.next(32, "test")
            val_loss = test(
                step_idx,
                test_data,
                mod,
                inner_opt_builder,
                num_inner_steps,
            )
            test_losses.append(val_loss)
            
# plot losses            
plt.plot(training_losses, label = 'Training Loss')
plt.plot(test_losses, label = 'Test Loss')
plt.xlabel('Training Iteration')
plt.ylabel('MSE')
plt.legend()
plt.show()
plt.close()

# view U matrix
U = mod.warp.detach().numpy()
U = U.reshape((70, 70, 3))

g_ind = 0
rho_g = U[g_ind]
rho_g_chopped = np.vstack([rho_g[0:5], rho_g[-5:]])

fig = plt.figure()
fig.set_size_inches(3,3)
plt.matshow(pi_g_chopped, aspect = 'auto', cmap = 'binary')
plt.colorbar()
plt.gca().axis('off')
plt.show()
plt.close()            