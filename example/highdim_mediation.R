# high-dimensional mediation analysis
library(tidyverse)
library(firatheme)
library(ggrepel)
library(lavaan)
library(tensorsem)

# params
DTYPE  <- torch_float64()
DEVICE <- torch_device("cuda")
LRATE  <- 0.05
TOL    <- 1e-20
MAXIT  <- 10000


# create data
med_df <-
  read_rds("https://github.com/vankesteren/sem-computationgraphs/raw/master/mediation_hidim/med_dat.rds") |>
  mutate_all(function(x) c(scale(x)))
med_dat <- torch_tensor(
  data = as.matrix(med_df),
  dtype = DTYPE,
  device = DEVICE,
  requires_grad = FALSE
)

# create model syntax
med_syntax <- paste0(
  paste(colnames(med_df[,-1:-2]), collapse = " + "), " ~ x\n",
  "y ~ ", paste(colnames(med_df[,-1:-2]), collapse = " + "), " + x"
)

# ULS estimation
N <- med_dat$shape[1]
P <- med_dat$shape[2]
s <- torch_vech(med_dat$t()$mm(med_dat) / N)

# create models & optim for training loop
med_mod_uls <- torch_sem(med_syntax, dtype = DTYPE, device = DEVICE)
optim <- optim_adam(med_mod_uls$parameters, lr = LRATE)
loss_uls <- numeric(MAXIT)

for (epoch in 1:MAXIT) {
  optim$zero_grad()
  r <- s - torch_vech(med_mod_uls())
  loss <- r$t()$mm(r)
  loss_uls[epoch] <- loss$item()
  if (epoch %% 10 == 1) {
    cat("\rEpoch:", epoch, " loss:", loss$item())
    flush.console()
  }
  loss$backward()
  optim$step()
  if (epoch > 1 && abs(loss_uls[epoch] - loss_uls[epoch - 1]) < TOL) {
    cat("\n")
    break
  }
}

loss_uls <- loss_uls[1:epoch]
plot(x = 1:epoch, y = loss_uls, xlab = "Epoch", ylab = "Loss value (ULS)", main = "Mediation model optimization", type = "l")
pt_uls <- lavMatrixRepresentation(med_mod_uls$partable(se = FALSE))

# save est as start vals for LASSO
uls_state <- med_mod_uls$state_dict()


# LASSO estimation
med_mod_lasso <- torch_sem(med_syntax, dtype = DTYPE, device = DEVICE)
med_mod_lasso$load_state_dict(uls_state)
optim <- optim_adam(med_mod_lasso$parameters, lr = 0.001)
loss_lasso <- numeric(MAXIT)
for (epoch in 1:MAXIT) {
  optim$zero_grad()
  r <- s - torch_vech(med_mod_lasso())
  paths <- torch_cat(c(med_mod_lasso$B_0[1001,1:1000], med_mod_lasso$B_0[1:1000, 1002]))
  loss <- r$t()$mm(r) + paths$abs()$sum()
  loss_lasso[epoch] <- loss$item()
  if (epoch %% 10 == 1) {
    cat("\rEpoch:", epoch, " loss:", loss$item())
    flush.console()
  }
  loss$backward()
  optim$step()
  if (epoch > 1 && abs(loss_lasso[epoch] - loss_lasso[epoch - 1]) < TOL) {
    cat("\n")
    break
  }
}
loss_lasso <- loss_lasso[1:epoch]
plot(x = 1:epoch, y = loss_lasso, xlab = "Epoch", ylab = "Loss value (LASSO)", main = "Mediation model optimization", type = "l")
pt_lasso <- lavMatrixRepresentation(med_mod_lasso$partable(se = FALSE))




tibble(
  "LASSO Estimate" = pt_lasso[1:1000, "est"]*pt_lasso[1001:2000, "est"],
  "ULS Estimate" = pt_uls[1:1000, "est"]*pt_uls[1001:2000, "est"],
  mediator = pt_lasso[1:1000, "lhs"],
  rowid = 1:1000
) %>%
  pivot_longer(-c(mediator, rowid)) %>%
  mutate(label = ifelse(name == "LASSO Estimate" & abs(value) > 0.006, mediator, "")) %>%
  ggplot(aes(x = rowid, y = abs(value), colour = as_factor(name), shape = as_factor(name), alpha = as_factor(name))) +
  geom_hline(yintercept = 0) +
  geom_point() +
  geom_text_repel(aes(label = label), color = "black") +
  scale_colour_fira() +
  scale_alpha_manual(values = c("LASSO Estimate" = 1, "ULS Estimate" = 0.5), guide = "none") +
  theme_fira() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Mediator", y = "Absolute indirect effect",
       title = "High-dimensional sparse mediation analysis: Gene methylation",
       colour = "", shape = "") +
  theme(legend.position = "top")















# now with majorization-minimization trick
med_mod_lasso_mm <- torch_sem(med_syntax, dtype = DTYPE, device = DEVICE)
med_mod_lasso_mm$load_state_dict(uls_state)
optim <- optim_adam(med_mod_lasso_mm$parameters, lr = LRATE)
loss_lasso_mm <- numeric(MAXIT)
for (epoch in 1:MAXIT) {
  optim$zero_grad()
  r <- s - torch_vech(med_mod_lasso_mm())
  paths <- torch_cat(c(med_mod_lasso_mm$B_0[1001,1:1000], med_mod_lasso_mm$B_0[1:1000, 1002]))
  loss <- r$t()$mm(r) + paths$div(paths$abs() + 1e-20)$dot(paths)
  loss_lasso_mm[epoch] <- loss$item()
  cat("\rEpoch:", epoch, " loss:", loss$item())
  flush.console()
  loss$backward()
  optim$step()
  if (epoch > 1 && abs(loss_lasso_mm[epoch] - loss_lasso_mm[epoch - 1]) < TOL) {
    cat("\n")
    break
  }
}
loss_lasso_mm <- loss_lasso_mm[1:epoch]
plot(x = 1:epoch, y = loss_lasso_mm, xlab = "Epoch", ylab = "Loss value (LASSO, MM)", main = "Mediation model optimization", type = "l")

pt_lasso_mm <- lavMatrixRepresentation(med_mod_lasso_mm$partable(se = FALSE))


tibble(
  "LASSO Estimate" = pt_lasso[1:1000, "est"]*pt_lasso[1001:2000, "est"],
  "MM LASSO Estimate" = pt_lasso_mm[1:1000, "est"]*pt_lasso_mm[1001:2000, "est"],
  "ULS Estimate" = pt_uls[1:1000, "est"]*pt_uls[1001:2000, "est"],
  mediator = pt_lasso[1:1000, "lhs"],
  rowid = 1:1000
) %>%
  pivot_longer(-c(mediator, rowid)) %>%
  mutate(label = ifelse(name == "LASSO Estimate" & abs(value) > 0.06, mediator, "")) %>%
  ggplot(aes(x = rowid, y = abs(value), colour = as_factor(name), shape = as_factor(name), alpha = as_factor(name))) +
  geom_hline(yintercept = 0) +
  geom_point() +
  geom_text_repel(aes(label = label), color = "black") +
  scale_colour_fira() +
  scale_alpha_manual(values = c("LASSO Estimate" = 1, "ULS Estimate" = 0.5), guide = "none") +
  theme_fira() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Mediator", y = "Absolute indirect effect",
       title = "High-dimensional sparse mediation analysis: Gene methylation",
       colour = "", shape = "") +
  theme(legend.position = "top")

