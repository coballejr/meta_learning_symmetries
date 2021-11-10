import math
import torch
from torch.nn import Module, init
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class ShareLinearFull(Module):
    '''
    adpated from https://github.com/AllanYangZhou/metalearning-symmetries/blob/master/synthetic_loader.py
    '''
    def __init__(self, in_features, out_features, bias=True, latent_size=3):
        super(ShareLinearFull, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.latent_params = Parameter(torch.Tensor(latent_size))
        self.warp = Parameter(torch.Tensor(in_features * out_features, latent_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def get_weight(self):
        return (self.warp @ self.latent_params).view(self.out_features, self.in_features)

    def reset_parameters(self):
        init._no_grad_normal_(self.warp, 0, 0.01)
        init._no_grad_normal_(self.latent_params, 0, 1 / self.out_features)
        if self.bias is not None:
            weight = self.get_weight()
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight = self.get_weight()
        return F.linear(x, weight, self.bias)
    
