# example analysis for SEM using pytorch
library(tensorsem)

# create a lavaan model for holzinger-swineford data
mod <- "
  # three-factor model
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"

# create SEM torch options from this model
opts <- syntax_to_torch_opts(mod)

# save the SEM torch options to a file
torch_opts_to_file(opts, filename = "example/hs_mod.pkl")

# save the holzinger-swineford data to a file
write.csv(lavaan::HolzingerSwineford1939, "example/hs.csv", row.names = FALSE)


# now run tensorsem_example.py to see the optimization and to return parameter estimates

# get parameter estimates from file and compare with lavaan
pt_torch  <- partable_from_torch(read.csv("example/pars.csv"), mod)
pt_lavaan <- parameterestimates(lavaan::sem(mod, HolzingerSwineford1939, std.lv = TRUE,
                                            information = "observed",
                                            fixed.x = FALSE))

# Estimates
cbind(pt_torch$est, pt_lavaan$est)

# Standard errors
cbind(pt_torch$se, pt_lavaan$se)
