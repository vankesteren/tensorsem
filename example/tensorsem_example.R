# example analysis for SEM using pytorch
library(tensorsem)

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
opts <- syntax_to_torch_opts(mod)
tsem <- torch_sem(opts, dtype = DTYPE)

# data needs to be centered (torch_sem does not support mean structure)
dat <- torch_tensor(scale(HolzingerSwineford1939[,7:15], scale = FALSE), requires_grad = FALSE)

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
  if (epoch %% 20 == 1) cat("Epoch:", epoch, " loss:", ll_values[epoch], "\n")
}

plot(1:epoch, -ll_values[1:epoch], type = "l", xlab = "epoch", ylab = "log-likelihood")


# Compare results with lavaan
fit_lavaan <- sem(
  model = mod,
  data = HolzingerSwineford1939,
  std.lv = TRUE,
  information = "observed",
  fixed.x = FALSE
)
pt_lavaan <- parameterestimates(fit_lavaan, remove.nonfree = TRUE)

# log-likelihood
-loss$item()
logLik(fit_lavaan)

# Estimates & standard errors
est <- as_array(tsem$free_params)

loss <- mvn_negloglik(dat, tsem())
hess <- tsem$Inverse_Hessian(loss)
ses  <- hess |> torch_diag() |> torch_sqrt() |> as_array()

# compare
round(cbind(torch_est = est, torch_se = ses, lav_est = pt_lavaan$est, lav_se = pt_lavaan$se), 3)
