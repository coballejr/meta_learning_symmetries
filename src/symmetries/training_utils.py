import numpy as np
import torch
import scipy.stats as st
import higher
import torch.nn.functional as F
import collections
from torch.optim import SGD, Adam

'''
adpated from https://github.com/AllanYangZhou/metalearning-symmetries/blob/master/synthetic_loader.py
'''

class SyntheticLoader:

    def __init__(self, data, k_spt=1, k_qry=19):
        self.xs, self.ys, self.ws = data["x"], data["y"], data["w"]
        # xs shape: (10000, 20, c_i, ...)
        # ys shape: (10000, 20, c_o, ...)
        self.c_i, self.c_o = self.xs.shape[2], self.ys.shape[2]
        self.k_spt, self.k_qry = k_spt, k_qry
        assert k_spt + k_qry <= 20, "Max 20 k_spt + k_20"
        train_cutoff = int(0.8 * self.xs.shape[0])
        self.train_range = range(train_cutoff)
        self.test_range = range(train_cutoff, self.xs.shape[0])

    def next(self, n_tasks, mode="train"):
        rnge = self.train_range if mode == "train" else self.test_range
        task_idcs = np.random.choice(rnge, n_tasks, replace=False)
        xs, ys, ws = self.xs[task_idcs], self.ys[task_idcs], self.ws[task_idcs]
        num_examples = xs.shape[1]
        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        for i in range(n_tasks):
            example_idcs = np.random.choice(num_examples, self.k_spt + self.k_qry, replace=False)
            spt_idcs, qry_idcs = example_idcs[: self.k_spt], example_idcs[self.k_spt :]
            x_spt.append(xs[i][spt_idcs])
            y_spt.append(ys[i][spt_idcs])
            x_qry.append(xs[i][qry_idcs])
            y_qry.append(ys[i][qry_idcs])
        x_spt = np.stack(x_spt)
        y_spt = np.stack(y_spt)
        x_qry = np.stack(x_qry)
        y_qry = np.stack(y_qry)
        data = [x_spt, y_spt, x_qry, y_qry]
        data = [torch.tensor(x) for x in data]
        return data, ws
    
def train(step_idx, data, net, inner_opt_builder, meta_opt, n_inner_iter):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    #querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt
    
    qry_losses = []
    meta_opt.zero_grad()
    for i in range(task_num):
        with higher.innerloop_ctx(
            net,
            inner_opt,
            copy_initial_weights=False,
            override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            qry_pred = fnet(x_qry[i])
            qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu().numpy())
            qry_loss.backward()
    #metrics = {"train_loss": np.mean(qry_losses)}
    meta_opt.step()
    
    return np.mean(qry_losses)


def test(step_idx, data, net, inner_opt_builder, n_inner_iter):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    #querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    for i in range(task_num):
        with higher.innerloop_ctx(
            net, inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            qry_pred = fnet(x_qry[i])
            qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu().numpy())
    avg_qry_loss = np.mean(qry_losses)
    _low, high = st.t.interval(
        0.95, len(qry_losses) - 1, loc=avg_qry_loss, scale=st.sem(qry_losses)
    )
    #test_metrics = {"test_loss": avg_qry_loss, "test_err": high - avg_qry_loss}
    return avg_qry_loss


def is_warp_layer(name):
    return "warp" in name


NAME_TO_INNER_OPT_CLS = {
    "maml": SGD,
    "maml_adam": Adam,
}


class InnerOptBuilder:
    def __init__(self, network, opt_name, init_lr, init_mode, lr_mode, ext_metaparams=None):
        self.network = network
        self.opt_name = opt_name
        self.init_lr = init_lr
        self.init_mode = init_mode
        self.lr_mode = lr_mode
        # metaparams that are not neural network params (e.g., learned lrs)
        if ext_metaparams:
            self.ext_metaparams = ext_metaparams
        else:
            self.ext_metaparams = self.make_ext_metaparams()
        self.inner_opt_cls = NAME_TO_INNER_OPT_CLS[opt_name]
        self.inner_opt = NAME_TO_INNER_OPT_CLS[opt_name](self.param_groups, lr=self.init_lr)

    def make_ext_metaparams(self):
        ext_metaparams = {}
        for name, param in self.network.named_parameters():
            if is_warp_layer(name) or not param.requires_grad:
                # Ignore symmetry params in the inner loop.
                continue
            if self.lr_mode == "per_layer":
                inner_lr = torch.tensor(self.init_lr)
                inner_lr.requires_grad = True
                ext_metaparams[f"{name}_lr"] = inner_lr
            elif self.lr_mode == "per_param":
                inner_lr = self.init_lr * torch.ones_like(param)
                inner_lr.requires_grad = True
                ext_metaparams[f"{name}_lr"] = inner_lr
            elif self.lr_mode == "fixed":
                pass
            else:
                raise ValueError(f"Unrecognized lr_mode: {self.lr_mode}")
        return ext_metaparams

    @property
    def metaparams(self):
        metaparams = {}
        metaparams.update(self.ext_metaparams)
        for name, param in self.network.named_parameters():
            if is_warp_layer(name) or self.init_mode == "learned":
                metaparams[name] = param
        return metaparams

    @property
    def param_groups(self):
        param_groups = []
        for name, param in self.network.named_parameters():
            if is_warp_layer(name) or not param.requires_grad:
                # Ignore symmetry params in the inner loop.
                continue
            param_groups.append({"params": param})
        return param_groups

    @property
    def overrides(self):
        overrides = collections.defaultdict(list)
        for name, param in self.network.named_parameters():
            if is_warp_layer(name) or not param.requires_grad:
                # Ignore symmetry params in the inner loop.
                continue
            if self.lr_mode == "per_layer":
                overrides["lr"].append(self.ext_metaparams[f"{name}_lr"])
            elif self.lr_mode == "per_param":
                overrides["lr"].append(self.ext_metaparams[f"{name}_lr"])
            elif self.lr_mode == "fixed":
                pass
            else:
                raise ValueError(f"Unrecognized lr_mode: {self.lr_mode}")
        return overrides
    
class CnLoader:
    
    def __init__(self, H,F,num_examples):
        self.H = H[0:num_examples, :]
        self.F = F[0:num_examples, :]
        self.num_examples = num_examples
        
    def next(self, batch_size):
        rand_idx = np.random.choice(self.num_examples, batch_size)
        H_batch, F_batch = self.H[rand_idx,:], self.F[rand_idx,:]
        H_batch, F_batch = torch.tensor(H_batch), torch.tensor(F_batch)
        return H_batch, F_batch
        
    