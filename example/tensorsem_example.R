# example analysis for SEM using pytorch
library(tensorsem)

# create a lavaan model for holzinger-swineford data
syntax <- "
  # three-factor model
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"

# create SEM torch model
mod <- torch_sem(syntax, dtype = torch_float64())

# data needs to be centered (torch_sem does not support mean structure)
dat <- torch_tensor(scale(HolzingerSwineford1939[,7:15], scale = FALSE), requires_grad = FALSE, dtype = torch_float64())

# fit torch sem model using maximum likelihood
mod$fit(dat)

# Compare results with lavaan
fit_lavaan <- sem(
  model = mod,
  data = HolzingerSwineford1939,
  std.lv = TRUE,
  information = "observed",
  fixed.x = FALSE
)
pt_lavaan <- parameterestimates(fit_lavaan)

# log-likelihood
logLik(fit_lavaan)

# Estimates & standard errors
ll <- mvn_negloglik(dat, tsem())
pt_torch <- tsem$partable(ll)

# compare
cbind(pt_torch, lav_est = pt_lavaan$est, lav_se = pt_lavaan$se)
