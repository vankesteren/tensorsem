# -*- coding: utf-8 -*-
# Example of using the tensorsem module
import torch
import pandas as pd
from tensorsem import *
import matplotlib.pyplot as plt
from pathlib import Path
import sys

### DESCRIPTION ###
# We will run the famous holzinger-swineford model from lavaan:
# a 3-factor confirmatory factor analysis model
# This model needs an options file (hs_mod.pkl)
# as well as a dataset (hs.csv).

### PARAMETERS ###
WORK_DIR = Path("example")  # the working directory
LRATE = 0.01  # Adam learning rate
TOL = 1e-20  # loss change tolerance
MAXIT = 5000  # maximum epochs
DTYPE = torch.float64  # 64-bit precision


### LOAD SEM OPTIONS AND DATASET ###
opts = SemOptions.from_file(WORK_DIR / "hs_mod.pkl")  # SemOptions is a special tensorsem settings class
df = pd.read_csv(WORK_DIR / "hs.csv")[opts.ov_names]  # order the columns, important step!
df -= df.mean(0)  # center the data
N, P = df.shape

dat = torch.tensor(df.values, dtype = DTYPE, requires_grad = False)

### MVN LOG-LIKELIHOOD OPTIMIZATION ###
model = StructuralEquationModel(opt = opts, dtype = DTYPE)  # instantiate the tensorsem model as a torch nn module
optim = torch.optim.Adam(model.parameters(), lr = LRATE)  # init the optimizer
loglik_values = []  # record loglik history in this list
for epoch in range(MAXIT):
    if epoch % 100 == 1:
        print("Epoch:", epoch, " loss:", loglik_values[-1])
    optim.zero_grad()  # reset the gradients of the parameters
    Sigma = model()  # compute the model-implied covariance matrix
    loss = mvn_negloglik(dat, Sigma)  # compute the negative log-likelihood
    loglik_values.append(-loss.item())  # record the log-likelihood
    loss.backward()  # compute the gradients and store them in the parameter tensors
    optim.step()  # take a step in the negative gradient direction using adam
    if epoch > 1:
        if abs(loglik_values[-1] - loglik_values[-2]) < TOL:
            break  # stop if no loglik change

# inspect optimization convergence
plt.plot(loglik_values)
plt.close()

# Inspecting the parameters
model.Lam  # Factor loadings matrix
model.Psi  # Factor covariance matrix
model.B_0  # Structural parameter matrix
model.Tht  # Residual covariance matrix

plt.imshow(model.Sigma.detach())  # plot implied covariance matrix
plt.close()

# Computing standard errors and p-values
loss = mvn_negloglik(dat, model())  # compute the likelihood at the optimum
se = model.Inverse_Hessian(loss).diag().sqrt()  # compute the standard errors from the observed information matrix
est = model.free_params  # get the free parameters
pval = 1 - torch.distributions.normal.Normal(0, 1).cdf(est/se)  # compute the p-value using a standard normal reference

# save them to a file
pd.DataFrame({"est": est.detach(), "se": se.detach()}).to_csv(WORK_DIR / "pars.csv")


### UNWEIGHTED LEAST SQUARES ESTIMATION ###
S = dat.t().mm(dat).div(N)  # N-normalized covariance matrix
s = vech(S)  # half-vectorized observed covariances
uls_model = StructuralEquationModel(opts)  # instantiate the tensorsem model as a torch nn module
optim = torch.optim.Adam(uls_model.parameters(), lr = LRATE)  # init the optimizer
uls_loss_values = []  # record losses in this list
for epoch in range(MAXIT):
    if epoch % 100 == 1:
        print("Epoch:", epoch, " loss:", uls_loss_values[-1])
    optim.zero_grad()  # reset the gradients of the parameters
    sigma = vech(uls_model())  # compute the model-implied covariance elements
    r = s - sigma  # compute the residual
    uls_loss = r.t().mm(r)  # the uls loss
    uls_loss_values.append(uls_loss.item())  # record the loss
    uls_loss.backward()  # compute the gradients and store them in the parameter tensors
    optim.step()  # take a step in the negative gradient direction using adam
    if epoch > 1:
        if abs(uls_loss_values[-1] - uls_loss_values[-2]) < TOL:
            break  # stop if no loss change

# inspect optimization convergence
plt.plot(uls_loss_values)
plt.close()

# Inspecting the parameters
uls_model.Lam  # Factor loadings matrix
uls_model.Psi  # Factor covariance matrix
uls_model.B_0  # Structural parameter matrix
uls_model.Tht  # Residual covariance matrix




### ASYMPTOTIC DISTRIBUTION-FREE DWLS ESTIMATION ###
Gamma = Gamma_ADF(dat)  # compute ADF gamma matrix
w = 1 / Gamma.diag()  # these are the dwls weights
dwls_model = StructuralEquationModel(opts)  # instantiate the tensorsem model as a torch nn module
optim = torch.optim.Adam(dwls_model.parameters(), lr = LRATE)  # init the optimizer
dwls_loss_values = []  # record losses in this list
for epoch in range(MAXIT):
    if epoch % 100 == 1:
        print("Epoch:", epoch, " loss:", dwls_loss_values[-1])
    optim.zero_grad()  # reset the gradients of the parameters
    sigma = vech(dwls_model())  # compute the model-implied covariance elements
    r = s - sigma  # compute the residual
    dwls_loss = r.t().mul(w).mm(r)  # the dwls loss
    dwls_loss_values.append(dwls_loss.item())  # record the loss
    dwls_loss.backward()  # compute the gradients and store them in the parameter tensors
    optim.step()  # take a step in the negative gradient direction using adam
    if epoch > 1:
        if abs(dwls_loss_values[-1] - dwls_loss_values[-2]) < TOL:
            break  # stop if no loss change

plt.plot(dwls_loss_values)
plt.close()

# Inspecting the parameters
dwls_model.Lam  # Factor loadings matrix
dwls_model.Psi  # Factor covariance matrix
dwls_model.B_0  # Structural parameter matrix
dwls_model.Tht  # Residual covariance matrix

# exit this script
sys.exit(0)