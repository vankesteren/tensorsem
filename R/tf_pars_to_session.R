#' Initialise a tensorflow environment from tf params
#'
#' @importFrom tensorflow tf shape
#'
#' @keywords internal
tf_pars_to_session <- function(params) {

  tf_env <- new.env()

  with(tf_env, {

    dat       <- create_tf_data(params$data_mat)

    # info
    v_trans   <- params$cov_map$v_trans
    v_itrans  <- params$cov_map$v_itrans
    v_names   <- params$cov_map$v_names



    # Parameter vector
    dlt_init  <- tf$constant(params$delta_start, dtype = "float32")
    dlt_free  <- tf$constant(params$delta_free,  dtype = "float32")
    dlt_value <- tf$constant(params$delta_value, dtype = "float32")
    dlt_vec   <- tf$Variable(initial_value =          dlt_init * dlt_free + dlt_value,
                             constraint    = function(dlt) dlt * dlt_free + dlt_value, dtype = "float32")

    vec_sizes <- tf$constant(vapply(params$idx, length, 1L), dtype = "int64")
    dlt_split <- tf$split(dlt_vec, vec_sizes)

    psi_vec <- dlt_split[[1]]
    b_0_vec <- dlt_split[[2]]
    lam_vec <- dlt_split[[3]]
    tht_vec <- dlt_split[[4]]

    free_split <- tf$split(dlt_free, vec_sizes)
    psi_free <- free_split[[1]]
    b_0_free <- free_split[[2]]
    lam_free <- free_split[[3]]
    tht_free <- free_split[[4]]

    # Psi matrix
    if (params$mat_size$psi[1] > 1L) {
      psi_dup <- tf$constant(matrixcalc::duplication.matrix(params$mat_size$psi[1]), dtype = "float32")
      psi_cc  <- tf$matmul(psi_dup, tf$expand_dims(psi_vec, 1L))
      Psi     <- tf$reshape(psi_cc, shape(params$mat_size$psi[1], params$mat_size$psi[2]))
    } else {
      Psi     <- tf$reshape(psi_vec, shape(params$mat_size$psi[1], params$mat_size$psi[2]))
    }

    # Beta matrix
    if (params$mat_size$beta[1] < 2) {
      B_0     <- tf$reshape(b_0_vec, shape(params$mat_size$beta[1], params$mat_size$beta[2]))
    } else {
      b_0_com <- tf$constant(matrixcalc::commutation.matrix(params$mat_size$beta[1], params$mat_size$beta[2]),
                             dtype = "float32")
      B_0     <- tf$reshape(tf$matmul(b_0_com, tf$expand_dims(b_0_vec, 1L)),
                            shape(params$mat_size$beta[1], params$mat_size$beta[2]))
    }

    # Lambda matrix
    if (params$mat_size$lambda[2] < 2) {
      Lambda  <- tf$reshape(lam_vec, shape(params$mat_size$lambda[1], params$mat_size$lambda[2]))
    } else {
      lam_com <- tf$constant(matrixcalc::commutation.matrix(params$mat_size$lambda[1], params$mat_size$lambda[2]),
                             dtype = "float32")
      Lambda  <- tf$reshape(tf$matmul(lam_com, tf$expand_dims(lam_vec, 1L)),
                            shape(params$mat_size$lambda[1], params$mat_size$lambda[2]))
    }

    # Theta matrix
    tht_dup   <- tf$constant(matrixcalc::duplication.matrix(params$mat_size$theta[1]), dtype = "float32")
    tht_cc    <- tf$matmul(tht_dup, tf$expand_dims(tht_vec, 1L))

    Theta     <- tf$reshape(tht_cc, shape(params$mat_size$theta[1],
                                          params$mat_size$theta[2]))

    # Get Sigma
    I_mat     <- tf$eye(params$mat_size$beta[1], dtype = "float32")
    B         <- I_mat - B_0
    B_inv     <- tf$matrix_inverse(B)
    Sigma     <- tf$matmul(tf$matmul(Lambda, tf$matmul(tf$matmul(B_inv, Psi), B_inv, transpose_b = TRUE)),
                           Lambda, transpose_b = TRUE) + Theta

    # Data batch
    N         <- tf$constant(dat$b_size, dtype = "float32")
    Z         <- dat$next_batch
    S         <- tf$matmul(Z$x, Z$x, transpose_a = TRUE) / N

    # Loss function
    loss      <- (tf$linalg$logdet(Sigma) + tf$linalg$trace(tf$matmul(S, tf$matrix_inverse(Sigma)))) * N / 2

    # abs diff gets really close to original estimates without inversion!
    # loss      <- tf$reduce_sum(tf$abs(Sigma - S))

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
    polyak    <- tf$train$ExponentialMovingAverage(decay = .998)
    polyak_v  <- ema$apply(tensorflow::tuple(B_0, Psi, Lambda, Theta))

    with(tf$control_dependencies(tensorflow::tuple(polyak_v)), {
      optim  <- tf$train$AdamOptimizer()
      train  <- optim$minimize(loss)
    })


    session <- tf$Session()
    session$run(tf$global_variables_initializer())

  })

  return(tf_env)
}
