dup_idx <- function(n) {
  indices <- integer(n^2)
  cur_idx <- 0
  for (row in 0:(n-1)) {
    for (col in 0:(n-1)) {
      cur_idx <- cur_idx + 1
      if (row == col) indices[cur_idx] <- row * (2 * n - row + 1) / 2
      if (row < col) indices[cur_idx] <- row * (2 * n - row + 1) / 2 + col - row
      if (row > col) indices[cur_idx] <- col * (2 * n - col + 1) / 2 + row - col
    }
  }
  return(indices + 1)
}


mvn_negloglik <- function(dat, Sigma) {
  # Multivariate normal negative log-likelihood loss function for tensorsem nn module.
  # :param dat: The centered dataset as a tensor
  # :param Sigma: The model() implied covariance matrix
  # :return: Tensor scalar negative log likelihood
  mu <- torch_zeros(Sigma$shape[1], dtype = Sigma$dtype)
  mvn <- distr_multivariate_normal(loc = mu, covariance_matrix = Sigma)
  return(mvn$log_prob(dat)$mul(-1)$sum())
}

jacobian <- function(output, input) {
  # Computes jacobian of output wrt input
  # :param output: Tensor vector of size Po
  # :param input: Tensor vector of size Pi
  # :return: jacobian: Tensor of size Pi, Po
  jac <- torch_zeros(output$shape[1], input$shape[1], dtype = input$dtype)
  for (i in 1:output$shape[1])
    jac[i] <- autograd_grad(output[i], input, retain_graph = TRUE)[[1]]

  return(torch_t(jac))
}
