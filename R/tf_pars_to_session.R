#' Initialise a tensorflow environment from tf params
#'
#' @importFrom tensorflow tf shape
#'
#' @keywords internal
tf_pars_to_session <- function(params) {

  tf_env <- new.env()

  with(tf_env, {

    # penalties
    lasso_beta   <- tf$placeholder(dtype = "float32", shape = shape(), name = "lasso_beta")
    lasso_lambda <- tf$placeholder(dtype = "float32", shape = shape(), name = "lasso_lambda")
    lasso_psi    <- tf$placeholder(dtype = "float32", shape = shape(), name = "lasso_psi")
    ridge_beta   <- tf$placeholder(dtype = "float32", shape = shape(), name = "ridge_beta")
    ridge_lambda <- tf$placeholder(dtype = "float32", shape = shape(), name = "ridge_lambda")
    ridge_psi    <- tf$placeholder(dtype = "float32", shape = shape(), name = "ridge_psi")

    # initialise dataset for batch processing / SGD
    # dat       <- create_tf_data(params$data_mat)
    dat <- tf_data$new(params$data_mat)

    # info
    v_trans   <- params$cov_map$v_trans
    v_itrans  <- params$cov_map$v_itrans
    v_names   <- params$cov_map$v_names

    # Parameter vector
    dlt_init  <- tf$constant(params$delta_start, dtype = "float32", name = "dlt_init")
    dlt_free  <- tf$constant(params$delta_free,  dtype = "float32", name = "dlt_free")
    dlt_value <- tf$constant(params$delta_value, dtype = "float32", name = "dlt_value")
    dlt_vec   <- tf$Variable(initial_value =          dlt_init * dlt_free + dlt_value,
                             constraint    = function(dlt) dlt * dlt_free + dlt_value, dtype = "float32",
                             name          = "dlt_vec")

    vec_sizes <- tf$constant(vapply(params$idx, length, 1L), dtype = "int64", name = "vec_sizes")
    dlt_split <- tf$split(dlt_vec, vec_sizes, name = "split_delta")

    psi_vec <- dlt_split[[1]]
    b_0_vec <- dlt_split[[2]]
    lam_vec <- dlt_split[[3]]
    tht_vec <- dlt_split[[4]]

    free_split <- tf$split(dlt_free, vec_sizes, name = "split_free")
    psi_free <- free_split[[1]]
    b_0_free <- free_split[[2]]
    lam_free <- free_split[[3]]
    tht_free <- free_split[[4]]

    # Psi matrix
    if (params$mat_size$psi[1] > 1L) {
      psi_dup <- tf$constant(matrixcalc::duplication.matrix(params$mat_size$psi[1]), dtype = "float32")
      psi_cc  <- tf$matmul(psi_dup, tf$expand_dims(psi_vec, 1L))
      Psi     <- tf$reshape(psi_cc, shape(params$mat_size$psi[1], params$mat_size$psi[2]), name = "Psi")
    } else {
      Psi     <- tf$reshape(psi_vec, shape(params$mat_size$psi[1], params$mat_size$psi[2]), name = "Psi")
    }

    # Beta matrix
    if (params$mat_size$beta[1] < 2) {
      B_0     <- tf$reshape(b_0_vec, shape(params$mat_size$beta[1], params$mat_size$beta[2]), name = "B_0")
    } else {
      b_0_com <- tf$constant(matrixcalc::commutation.matrix(params$mat_size$beta[1], params$mat_size$beta[2]),
                             dtype = "float32")
      B_0     <- tf$reshape(tf$matmul(b_0_com, tf$expand_dims(b_0_vec, 1L)),
                            shape(params$mat_size$beta[1], params$mat_size$beta[2]), name = "B_0")
    }

    # Lambda matrix
    if (params$mat_size$lambda[2] < 2) {
      Lambda  <- tf$reshape(lam_vec, shape(params$mat_size$lambda[1], params$mat_size$lambda[2]), name = "Lambda")
    } else {
      lam_com <- tf$constant(matrixcalc::commutation.matrix(params$mat_size$lambda[1], params$mat_size$lambda[2]),
                             dtype = "float32")
      Lambda  <- tf$reshape(tf$matmul(lam_com, tf$expand_dims(lam_vec, 1L)),
                            shape(params$mat_size$lambda[1], params$mat_size$lambda[2]), name = "Lambda")
    }

    # Theta matrix
    tht_dup   <- tf$constant(matrixcalc::duplication.matrix(params$mat_size$theta[1]), dtype = "float32")
    tht_cc    <- tf$matmul(tht_dup, tf$expand_dims(tht_vec, 1L))

    Theta     <- tf$reshape(tht_cc, shape(params$mat_size$theta[1],
                                          params$mat_size$theta[2]), name = "Theta")

    # Get Sigma
    I_mat     <- tf$eye(params$mat_size$beta[1], dtype = "float32")
    B         <- I_mat - B_0
    B_inv     <- tf$matrix_inverse(B)
    Sigma_ful <- tf$matmul(tf$matmul(Lambda, tf$matmul(tf$matmul(B_inv, Psi), B_inv, transpose_b = TRUE)),
                           Lambda, transpose_b = TRUE) + Theta

    # Data batch
    N         <- tf$constant(dat$b_size, dtype = "float32")
    Z         <- dat$get_next
    Z_data    <- Z[[1]]$x
    mask      <- tf$gather(Z[[2]]$x, 0L)
    Z_full    <- tf$boolean_mask(Z_data, mask, axis = 1L)
    S         <- tf$matmul(Z_full, Z_full, transpose_a = TRUE) / N

    # Also mask Sigma for missing data pattern
    Sigma     <- tf$boolean_mask(tf$boolean_mask(Sigma_ful, mask, axis = 0L), mask, axis = 1L)
    Sigma_inv <- tf$matrix_inverse(Sigma)

    # penalties
    penalty <-
      lasso_beta   * tf$reduce_sum(tf$abs(B_0)) +
      lasso_lambda * tf$reduce_sum(tf$abs(Lambda)) +
      lasso_psi    * tf$reduce_sum(tf$abs(Psi)) +
      ridge_beta   * tf$reduce_sum(tf$square(B_0)) +
      ridge_lambda * tf$reduce_sum(tf$square(Lambda)) +
      ridge_psi    * tf$reduce_sum(tf$square(Psi))

    # fit function
    fit <- switch(params$fit_fun,
      ml  = (tf$linalg$logdet(Sigma) + tf$linalg$trace(tf$matmul(S, Sigma_inv))) * N / 2,
      lad = tf$reduce_sum(tf$abs(Sigma - S)) * N
    )

    loss <- fit + penalty

    # gradients
    if (params$mat_size$psi[1] > 1L) {
      Psi_g   <- tf$reshape(tf$matmul(psi_dup, tf$transpose(tf$gradients(loss, psi_vec))),
                            shape(params$mat_size$psi[1], params$mat_size$psi[2]))
    } else {
      Psi_g   <- tf$reshape(tf$gradients(loss, psi_vec),
                            shape(params$mat_size$psi[1], params$mat_size$psi[2]))
    }
    B_0_g     <- tf$reshape(tf$gradients(loss, b_0_vec),
                            shape(params$mat_size$beta[1], params$mat_size$beta[2]))
    Lambda_g  <- tf$reshape(tf$gradients(loss, lam_vec),
                            shape(params$mat_size$lambda[1], params$mat_size$lambda[2]))
    Theta_g   <- tf$reshape(tf$matmul(tht_dup, tf$transpose(tf$gradients(loss, tht_vec))),
                            shape(params$mat_size$theta[1], params$mat_size$theta[2]))

    # free parameter vector
    idx_dlt   <- tf$constant(as.integer(which(params$delta_free == 1) - 1L), dtype = "int32")
    dlt_fre   <- tf$gather(dlt_vec, idx_dlt)
    dlt_g     <- tf$gradients(loss, dlt_vec)
    dlt_H     <- tf$hessians(loss, dlt_vec)


    # polyak average _all the things_
    polyak    <- tf$train$ExponentialMovingAverage(decay = params$polyak_decay, zero_debias = TRUE)
    polyak_v  <- polyak$apply(tensorflow::tuple(
      B_0, B_0_g,
      Psi, Psi_g,
      Lambda, Lambda_g,
      Theta, Theta_g,
      Sigma_ful, loss,
      dlt_vec, dlt_fre, dlt_g[[1]], dlt_H[[1]]
    ))

    # initialise the optimizer
    with(tf$control_dependencies(tensorflow::tuple(polyak_v)), {
      optim  <- tf$train$AdamOptimizer()
      train  <- optim$minimize(loss)
    })
    reset_optim_op <- tf$variables_initializer(optim$variables())

    # initialise session
    session <- tf$Session()
    session$run(tf$global_variables_initializer())
  })

  return(tf_env)
}
