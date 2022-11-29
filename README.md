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
An `R`  package for structural equation modeling using `Torch`. This package is meant for researchers who know their way around SEM, Torch, and lavaan. 


> Note: `tensorsem` used to be a combined `R` and `python` package, with model specification in `R` and estimation in `python`. To view and download version 1, go to [`v1.0`](https://github.com/vankesteren/tensorsem/tree/v1.0)

Structural equation modeling is implemented as a fully functional `torch nn` module. A short example optimization loop would be:

```R
library(tensorsem)

# model syntax (we use lavaan)
syntax <- "
  # three-factor model
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"

# create a data object as a torch tensor
dat <- torch_tensor(
  data = scale(HolzingerSwineford1939[,7:15], scale = FALSE),
  requires_grad = FALSE
)

# initialize the SEM model object
model <- torch_sem(syntax) 

# initialize the optimizer
opt <- optim_adam(model$parameters, lr = 0.01)

# optimize
for (epoch in 1:500) {
  opt$zero_grad()  # reset the gradients of the parameters
  Sigma <- model()  # compute the model-implied covariance matrix
  
  # compute the negative log-likelihood
  dist <- distr_multivariate_normal(loc = model$mu, covariance_matrix = Sigma)
  loss <- -dist$log_prob(dat)$sum()
  loss$backward()  # compute gradients and store them in parameter tensors
  opt$step()  # take a step in the negative gradient direction using adam
}

# show parameter table
model$partable()
```

```
   id     lhs op     rhs user block group free ustart exo label plabel start   est se
1   1  visual =~      x1    1     1     1    1     NA   0         .p1.     1 0.900 NA
2   2  visual =~      x2    1     1     1    2     NA   0         .p2.     1 0.498 NA
3   3  visual =~      x3    1     1     1    3     NA   0         .p3.     1 0.656 NA
4   4 textual =~      x4    1     1     1    4     NA   0         .p4.     1 0.990 NA
5   5 textual =~      x5    1     1     1    5     NA   0         .p5.     1 1.102 NA
6   6 textual =~      x6    1     1     1    6     NA   0         .p6.     1 0.917 NA
7   7   speed =~      x7    1     1     1    7     NA   0         .p7.     1 0.619 NA
8   8   speed =~      x8    1     1     1    8     NA   0         .p8.     1 0.731 NA
9   9   speed =~      x9    1     1     1    9     NA   0         .p9.     1 0.670 NA
10 10      x1 ~~      x1    0     1     1   10     NA   0        .p10.     1 0.549 NA
11 11      x2 ~~      x2    0     1     1   11     NA   0        .p11.     1 1.134 NA
12 12      x3 ~~      x3    0     1     1   12     NA   0        .p12.     1 0.844 NA
13 13      x4 ~~      x4    0     1     1   13     NA   0        .p13.     1 0.371 NA
14 14      x5 ~~      x5    0     1     1   14     NA   0        .p14.     1 0.446 NA
15 15      x6 ~~      x6    0     1     1   15     NA   0        .p15.     1 0.356 NA
16 16      x7 ~~      x7    0     1     1   16     NA   0        .p16.     1 0.799 NA
17 17      x8 ~~      x8    0     1     1   17     NA   0        .p17.     1 0.488 NA
18 18      x9 ~~      x9    0     1     1   18     NA   0        .p18.     1 0.566 NA
19 19  visual ~~  visual    0     1     1    0      1   0        .p19.     1 1.000  0
20 20 textual ~~ textual    0     1     1    0      1   0        .p20.     1 1.000  0
21 21   speed ~~   speed    0     1     1    0      1   0        .p21.     1 1.000  0
22 22  visual ~~ textual    0     1     1   19     NA   0        .p22.     0 0.459 NA
23 23  visual ~~   speed    0     1     1   20     NA   0        .p23.     0 0.471 NA
24 24 textual ~~   speed    0     1     1   21     NA   0        .p24.     0 0.283 NA
```

## Installation
To install the latest version of `tensorsem`, run the following:

1. Install `torch` as per the instructions here: [https://github.com/mlverse/torch](https://github.com/mlverse/torch)
2. Install the `R` package from this repository:
    ```r
    remotes::install_github("vankesteren/tensorsem")
    ```

## Usage
See the [`examples`](./examples/) directory for some usage examples and documentation.
