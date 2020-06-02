# -*- coding: utf-8 -*-
# Loss functions to export
import torch

def mvn_negloglik(dat, Sigma):
    """
    Multivariate normal negative log-likelihood loss function for tensorsem nn module.
    :param dat: The centered dataset as a tensor
    :param Sigma: The model() implied covariance matrix
    :return: Tensor scalar negative log likelihood
    """
    mu = torch.zeros(Sigma.shape[0], dtype = Sigma.dtype)
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc = mu, covariance_matrix = Sigma)
    return mvn.log_prob(dat).mul(-1).sum()

def sem_fitfun(S, Sigma):
    Sigma_chol = Sigma.cholesky()
    # log-determinant + trace of S*sigma inv
    return 2 * Sigma_chol.diag().log().sum() + S.cholesky_solve(Sigma_chol).trace()
