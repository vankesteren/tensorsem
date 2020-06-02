# -*- coding: utf-8 -*-
# The tensorsem module
import torch
from .semopts import SemOptions
from .helpers import dup_idx
from .functions import jacobian

class StructuralEquationModel(torch.nn.Module):
    def __init__(self, opt: SemOptions, dtype: torch.dtype = torch.float32):
        """
        In The constructor we instantiate the parameter vector, its constraints,
        as well as several helper lists and tensors for creating the model from
        this parameter vector
        :param opt_dict: dictionary from lav_to_tf_pars
        :param dtype: device type, default float32
        """
        super(StructuralEquationModel, self).__init__()
        self.opt = opt
        # Initialize the parameter vector
        self.dlt_vec = torch.nn.Parameter(torch.tensor(opt.delta_start, dtype = dtype))

        # Initialize parameter constraints
        self.dlt_free = torch.tensor(opt.delta_free, dtype = torch.bool, requires_grad = False)
        self.dlt_value = torch.tensor(opt.delta_value, dtype = dtype, requires_grad = False)

        # duplication indices transforming vech to vec for psi and theta
        self.psi_dup_idx = torch.tensor(dup_idx(opt.psi_shape[0]), dtype = torch.long, requires_grad = False)
        self.tht_dup_idx = torch.tensor(dup_idx(opt.tht_shape[0]), dtype = torch.long, requires_grad = False)

        # tensor identity matrix
        self.I_mat = torch.eye(opt.b_0_shape[0], dtype = dtype, requires_grad = False)

    def forward(self):
        """
        In the forward pass, we apply constraints to the parameter vector, and we
        create matrix views from it to compute the model-implied covariance matrix
        :return: model-implied covariance matrix tensor
        """
        # Apply constraints
        self.dlt = self.dlt_vec.where(self.dlt_free, self.dlt_value)

        # Create the model matrix views of the delta vector
        self.dlt_split = self.dlt.split(self.opt.delta_sizes)

        # Create individual matrix views
        self.Lam = self.dlt_split[0].view(self.opt.lam_shape[1], self.opt.lam_shape[0]).t()  # fill matrix by column
        self.Tht = self.dlt_split[1].index_select(0, self.tht_dup_idx).view(self.opt.tht_shape)
        self.Psi = self.dlt_split[2].index_select(0, self.psi_dup_idx).view(self.opt.psi_shape)
        self.B_0 = self.dlt_split[3].view(self.opt.b_0_shape[1], self.opt.b_0_shape[0]).t()  # fill matrix by column

        # Now create model-implied covariance matrix
        self.B = self.I_mat - self.B_0
        self.B_inv = self.B.inverse()
        self.Sigma = self.Lam.mm(self.B_inv.mm(self.Psi).mm(self.B_inv.t())).mm(self.Lam.t()).add(self.Tht)
        return self.Sigma

    def Inverse_Hessian(self, loss):
        """
        Computes and returns the asymptotic covariance matrix of the parameters with
        respect to the loss function, to compute standard errors (sqrt(diag(ACOV)))
        :param loss: freshly computed loss function (for backwards pass)
        :return: ACOV tensor of the free parameters
        """
        g = torch.autograd.grad(loss, self.dlt_vec, create_graph = True)[0]
        H = jacobian(g, self.dlt_vec)
        free_idx = self.dlt_free.nonzero().view(-1)
        self.Hinv = H[free_idx, :][:, free_idx].inverse()
        return self.Hinv

    @property
    def free_params(self):
        """
        Returns the free parameter vector
        :return: Tensor with free parameters
        """
        return self.dlt_vec[self.dlt_free.nonzero().view(-1)]
