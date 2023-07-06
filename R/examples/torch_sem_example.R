# create a lavaan syntax for holzinger-swineford data
syntax <- "
  # three-factor model
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"

# create SEM torch model
model <- torch_sem(syntax)

# prepare dataset
dat <- df_to_tensor(HolzingerSwineford1939[,7:15])

# fit model
model$fit(dat)

# compute loss
loss <- -model$loglik(dat)

# show parameter table
model$partable(loss)
