#' Constructs index vector for transforming a vech vector
#' into a vec vector to create an n*n symmetric matrix
#' from the vech vector.
#' tensor$index_select(0, idx)$view(3,3)
#'
#' @param n size of the resulting square matrix
#' @return array containing the indices
vech_dup_idx <- function(n) {
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


#' Compute jacobian of output wrt input tensor
#'
#' @param output Tensor vector of size Po
#' @param input Tensor vector of size Pi
#'
#' @return jacobian: Tensor of size Pi, Po
torch_jacobian <- function(output, input) {
  jac <- torch_zeros(output$shape[1], input$shape[1], dtype = input$dtype)
  for (i in 1:output$shape[1])
    jac[i] <- autograd_grad(output[i], input, retain_graph = TRUE)[[1]]

  return(torch_t(jac))
}


#' Half-vectorization of square matrices
#'
#' @param x square (symmetric) matrix tensor
#'
#' @return column vector of stacked lower-diagonal elements
torch_vech <- function(x) {
  P <- x$shape[1]
  Ps <- round(P*(P+1)/2)
  idx_1d <- lavaan::lav_matrix_vech_idx(P)
  return(x$view(-1)[idx_1d]$view(c(Ps, 1)))
}
