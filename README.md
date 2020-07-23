<p align="center">
  <img src="img/tensorsem.png" width="300px"></img>
  <br/>
  <span>
    <a href="https://travis-ci.org/vankesteren/tensorsem"><img src="https://travis-ci.org/vankesteren/tensorsem.svg?branch=master"></img></a>
    <a href="https://zenodo.org/badge/latestdoi/168356695"><img src="https://zenodo.org/badge/168356695.svg" alt="DOI"></a>
  </span>
  <h5 align="center">Structural Equation Modeling using Torch</h5>
</p>
<br/>

## Description
An `R` and `python` package for structural equation modeling using `Torch`. This package is meant for researchers who know their way around SEM, Torch, and lavaan. 
Structural equation modeling is implemented as a fully functional `torch.nn` module. A short example optimization loop would be:

```python
import tensorsem as ts
model = ts.StructuralEquationModel(opt = opts)  # opts are of class SemOptions
optim = torch.optim.Adam(model.parameters())  # init the optimizer
for epoch in range(1000):
    optim.zero_grad()  # reset the gradients of the parameters
    Sigma = model()  # compute the model-implied covariance matrix
    loss = ts.mvn_negloglik(dat, Sigma)  # compute the negative log-likelihood, dat tensor should exist
    loss.backward()  # compute the gradients and store them in the parameter tensors
    optim.step()  # take a step in the negative gradient direction using adam
```

## Installation
To install the latest version of `tensorsem`, run the following:

1. Install the `R` interface package from this repository:
    ```r
    remotes::install_github("vankesteren/tensorsem")
    ```
2. Install `pytorch` on your system. Use the [`pytorch` website](https://pytorch.org/get-started/locally/) to do this. For example, for a windows pip cpu version, use:
    ```shell script
    pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    ```
3. Install the `tensorsem` `python` package from this repository.
    ```shell script
    pip install https://github.com/vankesteren/tensorsem/archive/master.zip
    ```
4. (Optional) Install `pandas` and `matplotlib` for plotting and parameter storing
    ```shell script
    pip install matplotlib pandas
    ```

## Usage
See the [example](example) directory for a full usage example, estimating the Holzinger-Swineford model using maximum likelihood, unweighted least squares, and diagonally weighted least squares.
