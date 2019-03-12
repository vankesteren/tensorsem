set.seed(45)
n      <- 100
p      <- 3
p_miss <- 0.2
S_true <- cov2cor(rWishart(1, p, diag(p))[,,1])
miss   <- matrix(rbinom(n*p, 1, p_miss), n)
X_full <- MASS::mvrnorm(n, rep(0, p), S_true)
X_miss <- X_full
X_miss[which(miss == 1)] <- NA

dataset <- data.frame(X_miss)
dataset_full <- data.frame(X_full)

library(lavaan)

lav <- "X1 ~ X2 + X3"
mod <- sem(lav, dataset, information = "observed", missing = "fiml")

tf_mod <- tf_sem(lav, dataset)
tf_mod$train(1000, verbose = FALSE)
