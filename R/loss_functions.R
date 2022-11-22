#' Multivariate normal negative log-likelihood loss function for tensorsem nn module.
#'
#' @param dat The centered dataset as a tensor
#' @param Sigma The model() implied covariance matrix
#'
#' @return torch_tensor: scalar negative log likelihood
mvn_negloglik <- function(dat, Sigma) {
  mu <- torch_zeros(Sigma$shape[1], dtype = Sigma$dtype)
  mvn <- distr_multivariate_normal(loc = mu, covariance_matrix = Sigma)
  return(mvn$log_prob(dat)$mul(-1)$sum())
}


#' SEM fitting function
#'
#' @param S The observed covariance matrix
#' @param Sigma The model implied covariance matrix
#'
#' @return torch_tensor: scalar loss function
sem_fitfun <- function(S, Sigma) {
  Sigma_chol <- linalg_cholesky(Sigma)

  # sem fitting function is log-determinant + trace of S*sigma inv
  logdet <- 2 * torch_sum(torch_log(torch_diag(Sigma_chol)))
  strace <- torch_trace(torch_cholesky_solve(S, Sigma_chol))
  return(logdet + strace)
}
