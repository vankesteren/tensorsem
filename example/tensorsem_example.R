# example analysis for SEM using pytorch
library(tensorsem)
library(torch)

# parameters
LRATE <- 0.01  # Adam learning rate
TOL   <- 1e-20  # loss change tolerance
MAXIT <- 5000  # maximum epochs
DTYPE <- torch_float64()  # 64-bit precision

# create a lavaan model for holzinger-swineford data
mod <- "
  # three-factor model
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"

# create SEM torch model
lav <- sem(mod, HolzingerSwineford1939, do.fit = FALSE)
opts <- syntax_to_torch_opts(mod)
tsem <- torch_sem(opts, dtype = DTYPE)

# data
dat <- torch_tensor(scale(lav@Data@X[[1]], scale = FALSE), requires_grad = FALSE)

# optimization loop
optim <- optim_adam(tsem$parameters, lr = LRATE)
ll_values <- numeric(MAXIT)

for (epoch in 1:MAXIT) {
  # reset gradients to 0
  optim$zero_grad()

  # compute the model-implied covariance matrix
  Sigma <- tsem()

  # compute the negative log-likelihood
  loss <- mvn_negloglik(dat, Sigma)

  # record loss function
  ll_values[epoch] <- loss$item()

  # compute the gradients and store them in the parameter tensors
  loss$backward()

  # take a step in the negative gradient direction using adam
  optim$step()

  # stop if no loglik change
  if (epoch > 1)
    if (abs(ll_values[epoch] - ll_values[epoch - 1]) < TOL)
      break

  # print loss to monitor
  if (epoch %% 100 == 1) cat("Epoch:", epoch, " loss:", ll_values[epoch], "\n")
}

plot(1:epoch, ll_values[1:epoch], type = "l")


# get parameter estimates from file and compare with lavaan
pt_torch  <- partable_from_torch(read.csv("example/pars.csv"), mod)
pt_lavaan <- parameterestimates(lavaan::sem(mod, HolzingerSwineford1939, std.lv = TRUE,
                                            information = "observed",
                                            fixed.x = FALSE))

# Estimates
cbind(pt_torch$est, pt_lavaan$est)

# Standard errors
cbind(pt_torch$se, pt_lavaan$se)
