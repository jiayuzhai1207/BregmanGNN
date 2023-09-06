
import numpy as np
import re
import torch
import torch.nn as nn

eps = 1e-5

# ReLU's smooth version & inverse
def bent_identity(var_in, param=1.):
    return (var_in + torch.sqrt(var_in ** 2 + param)) / (2 * param)

def bent_identity_conjugate(var_in, param=1.):
    return param * var_in + 1 / var_in

# ISRU
def isru(var_in):  # activation func: isru(x) = x / sqrt(1 + x^2)
    return torch.div(var_in, torch.sqrt(1 + var_in ** 2))

def isru_conjugate(var_in):
    return torch.div(var_in, torch.sqrt(1 - var_in ** 2))

# Sigmoid
def sigmoid_conjugate(var_in):
    return torch.log(torch.div(var_in, 1 - var_in))  # 改过

# Scaled atan
# Scaled tanh
def asin(var_in, threshold=1e-5):
    return torch.where(var_in < -1+threshold, np.pi/2-np.sqrt(2)*torch.sqrt(var_in+1),
                       torch.where(var_in > 1-threshold, np.pi/2 + np.sqrt(2)*torch.sqrt(1-var_in),
                       torch.asin(var_in)))
# asinh
def asinh(var_in):
    return torch.log(var_in + torch.sqrt(var_in ** 2 + 1))

# softplus
def log_sigmoid(var_in):
    min_elem = torch.min(var_in, torch.zeros_like(var_in))
    z = torch.exp(min_elem) + torch.exp(min_elem - var_in)
    return min_elem - torch.log(z)

def softplus(var_in, beta=1, threshold=10):
    """
    SoftPlus functional.
    - Option A: Standard implementation
    - Option B: Stable implementation without threshold from the following post
        [https://github.com/pytorch/pytorch/issues/31856]
    """
    option = 'B'
    if option == 'A':
        return torch.nn.functional.softplus(var_in, beta=beta, threshold=threshold)
    else:
        return - (1 / beta) * log_sigmoid(-beta * var_in)

def softplus_conjugate(var_in, beta=1, threshold=5):
    """
    Inverse of SoftPlus functional.
    - Option A: Standard implementation
    - Option B: Stable implementation inspired from the following TensorFlow code:
        [https://github.com/tensorflow/probability/blob/v0.15.0/tensorflow_probability/python/math/generic.py#L494-L545]
    """
    option = 'B'
    if option == 'A':
        return torch.log(torch.exp(beta * var_in) - 1) / beta
    else:
        # = (1 / beta) * torch.log(1 - torch.exp(-beta * var_in)) + var_in
        return torch.where(var_in * beta < np.exp(-threshold), torch.log(beta*var_in)/beta + var_in,
                           (1 / beta) * torch.log(1 - torch.exp(-beta * var_in)) + var_in)


class Zeros(nn.Module):
    def __init__(self):
        super(Zeros, self).__init__()

    @staticmethod
    def forward(var_input):
        return torch.zeros_like(var_input)

def get(activation_name, version, beta=1000):
    """ Get the couple (activation/offset) for Bregman, Euclidean and Standard neural networks """
    if activation_name == 'relu':
        activation = nn.ReLU()
        smooth_activation = bent_identity
        smooth_offset = bent_identity_conjugate
        v_range = [0, np.Inf]
    elif activation_name == 'isru':
        activation = isru
        smooth_activation = isru
        smooth_offset = isru_conjugate
        v_range = [-1, 1]
    elif activation_name == 'sigmoid':
        activation = nn.Sigmoid()
        smooth_activation = nn.Sigmoid()
        smooth_offset = sigmoid_conjugate
        v_range = [0, 1]
    elif activation_name == 'tanh':
        activation = torch.tanh
        smooth_activation = torch.tanh
        smooth_offset = torch.atanh
        v_range = [-1, 1]
    elif activation_name == 'scaled_tanh':
        a = 1.7159
        b = 0.6666
        activation = (lambda var_in: a*torch.tanh(b*var_in))
        smooth_activation = (lambda var_in: a*torch.tanh(b*var_in))
        smooth_offset = (lambda var_in: (1/b)*torch.atanh(var_in/a))
        v_range = [-a, a]
    elif activation_name == 'atan':
        activation = torch.atan
        smooth_activation = torch.atan
        smooth_offset = torch.tan
        v_range = [-np.pi / 2, np.pi / 2]
    elif activation_name == 'sin':
        activation = torch.sin
        smooth_activation = torch.sin
        smooth_offset = asin #torch.asin
        v_range = [-1, 1]
    elif activation_name == 'asinh':
        activation = asinh
        smooth_activation = asinh
        smooth_offset = torch.sinh
        v_range = [-np.Inf, np.Inf]
    elif 'softplus' in activation_name:
        beta = [float(s) for s in re.findall(r'[\d\.\d]+',activation_name)]
        if not beta:
            beta = 1000
        else:
            beta = beta[0]
        activation = (lambda var: softplus(var, beta=beta))
        smooth_activation = (lambda var: softplus(var, beta=beta))
        smooth_offset = (lambda var: softplus_conjugate(var, beta=beta))
        v_range = [0, np.Inf]
    else:
        activation = None
        smooth_activation = None
        smooth_offset = None
        v_range = None
        print('incorrect regularization')

    if version == 'bregman':
        v_range = [v_range[0] + eps, v_range[1] - eps]
        return smooth_activation, smooth_offset, v_range
    elif version == 'euclidean':
        return activation, torch.nn.Identity(), [-np.Inf, np.Inf]
    elif version == 'standard':
        return activation, Zeros(), [-np.Inf, np.Inf]

