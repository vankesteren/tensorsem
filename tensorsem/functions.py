# -*- coding: utf-8 -*-
# Helper functions to export for use in loss functions and SE computation
import torch

def Gamma_ADF(data):
    """
    Constructs gamma matrix from raw data based on asymptotic distribution free theory. Source:
    https://github.com/yrosseel/lavaan/blob/f630fe752f75ec6c22fdf57de4c940a1612381b5/R/lav_samplestats_gamma.R#L241
    :param data: N * P tensor of raw data
    :return: Gamma tensor
    """
    N, P = data.shape
    Y = data
    Yc = Y.sub(Y.mean(0)) # center
    flatten = lambda l: [item for sublist in l for item in sublist]
    idx1 = flatten([p] * (P - p) for p in range(P))
    idx2 = flatten([list(range(p, P)) for p in range(P)])
    Z = Yc[:,idx1] * Yc[:,idx2]
    Zc = Z.sub(Z.mean(0)) # center
    return Zc.t().mm(Zc).div(N)

def jacobian(output, input):
    """
    Computes jacobian of output wrt input
    :param output: Tensor vector of size Po
    :param input: Tensor vector of size Pi
    :return: jacobian: Tensor of size Pi, Po
    """
    jacobian = torch.zeros(output.shape[0], input.shape[0])
    for i in range(output.shape[0]):
        jacobian[i] = torch.autograd.grad(output[i], input, retain_graph = True)[0]

    return jacobian.t()

def vech(x: torch.Tensor):
    """
    :param x: square (symmetric) matrix tensor
    :return: column vech
    """
    P = x.shape[0]
    Ps = int(P*(P+1)/2)
    tridx = torch.tril_indices(P, P)
    return x[tridx[0, :], tridx[1, :]].view(Ps, 1)