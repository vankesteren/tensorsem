## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = FALSE,
  comment = "#>",
  cache = TRUE,
  out.width = 7,
  out.height = 4,
  warning = FALSE
)

## ----setup--------------------------------------------------------------------
library(tensorsem)

## ----lavaan-------------------------------------------------------------------
# Create model syntax
syntax <- "
  # three-factor model
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"

# Fit lavaan model
fit_lavaan <- sem(
  model = syntax,
  data = HolzingerSwineford1939,
  std.lv = TRUE,
  information = "observed",
  fixed.x = FALSE
)

pt_lavaan <- partable(fit_lavaan)

## ----paramplot----------------------------------------------------------------
param_plot <- function(...) {
  ptli <- list(...)
  ptli <- lapply(names(ptli), function(n) {
    out <- ptli[[n]]
    out$method <- n
    return(out)
  })
  Reduce(rbind, ptli) |> 
    ggplot2::ggplot(ggplot2::aes(x = id, y = est, colour = method,
               ymin = est - 1.96*se, ymax = est + 1.96*se)) + 
    ggplot2::geom_pointrange(position = ggplot2::position_dodge(width = .8)) +
    ggplot2::theme_minimal() +
    ggplot2::labs(x = "Parameter", y = "Value", colour = "Method")
}

param_plot(lavaan = pt_lavaan)

## ----tensorsem-ml-------------------------------------------------------------
# initialize the SEM model object
mod_ml <- torch_sem(syntax, dtype = torch_float64())

# create a data object as a torch tensor
dat_torch <- torch_tensor(
  data = scale(HolzingerSwineford1939[,7:15], scale = FALSE),
  requires_grad = FALSE,
  dtype = torch_float64(),
)

# estimate the model using default settings
mod_ml$fit(dat = dat_torch, verbose = FALSE)

# re-compute the log-likelihood & create partable
ll <- mod_ml$loglik(dat_torch)
pt_ml <- mod_ml$partable(-ll)

# compare to lavaan
param_plot(
  lavaan = pt_lavaan,
  torch_ml = pt_ml
)


## ----tdistribution------------------------------------------------------------
# Create a multivariate t log-likelihood distribution using torch operations
# for reference, see here: https://docs.pyro.ai/en/stable/_modules/pyro/distributions/multivariate_studentt.html
# note that that code is APACHE-2.0 licensed
mvt_loglik <- function(x, Sigma, nu = 2) {
  p <- x$shape[2]
  n <- x$shape[1]
  
  Schol <- linalg_cholesky(Sigma)

  # constant term
  C <- Schol$diag()$log()$sum() +  p/2*torch_log(nu) + p/2*log(pi) + torch_lgamma(nu / 2) - torch_lgamma((p + nu) / 2)
  
  # data-dependent term
  y <- x$t()$triangular_solve(Schol, upper = FALSE)[[1]]
  D <- torch_log1p(y$square()$sum(1) / nu)
  
  return(-0.5 * (nu + p) * D - C)
}

## ----mvt_opt------------------------------------------------------------------
# initialize the SEM model object
mod_t <- torch_sem(syntax, dtype = torch_float64())

# initialize at the ml estimates for faster convergence
mod_t$dlt_vec <- as_array(mod_ml$dlt_vec)

# initialize the optimizer
opt <- optim_adam(mod_t$parameters, lr = 0.01)

# start the training loop
iters <- 200L
loglik <- numeric(iters)
for (i in 1:iters) {
  opt$zero_grad()
  Sigma <- mod_t()
  loss <- -mvt_loglik(dat_torch, Sigma)$sum()
  loglik[i] <- -loss$item()
  loss$backward()
  opt$step()
}


## ----mvtplot------------------------------------------------------------------
opt_plot <- function(losses) {
  ggplot2::ggplot(data.frame(x = 1:length(losses), y = losses), ggplot2::aes(x, y)) + 
    ggplot2::geom_line() +
    ggplot2::theme_minimal()
}
opt_plot(loglik) +
  ggplot2::labs(x = "Epochs", y = "Log-Likelihood", title = "ML estimation of multivariate T SEM")

## ----mvtpars------------------------------------------------------------------
# re-compute the log-likelihood & create partable
ll <- mvt_loglik(dat_torch, mod_t())$sum()
pt_t <- mod_t$partable(-ll)

# compare to lavaan
param_plot(mvn = pt_ml, t = pt_t)

