#' loss functions
#'
#' @importFrom tensorflow tf
#'
#' @keywords internal
ml_loss <- expression(
  N / 2 *
    (tf$linalg$logdet(Sigma) +
     tf$linalg$trace(tf$matmul(S, tf$matrix_inverse(Sigma))))
)
