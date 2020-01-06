#' Initialise a tensorflow environment from tf params
#'
#' @importFrom tensorflow tf shape
#'
#' @keywords internal
tf_pars_to_session <- function(params) {

  tf_env <- new.env()

  with(tf_env, {
    tf1 <- tf$compat$v1
    tf$python$framework_ops$disable_eager_execution()

    # penalties
    lasso_beta     <- tf1$placeholder(dtype = "float32", shape = shape(), name = "lasso_beta")
    lasso_lambda   <- tf1$placeholder(dtype = "float32", shape = shape(), name = "lasso_lambda")
    lasso_psi      <- tf1$placeholder(dtype = "float32", shape = shape(), name = "lasso_psi")
    ridge_beta     <- tf1$placeholder(dtype = "float32", shape = shape(), name = "ridge_beta")
    ridge_lambda   <- tf1$placeholder(dtype = "float32", shape = shape(), name = "ridge_lambda")
    ridge_psi      <- tf1$placeholder(dtype = "float32", shape = shape(), name = "ridge_psi")

    # spike-slab params
    spike_lambda  <- tf1$placeholder(dtype = "float32", shape = shape(), name = "spike_lambda")
    slab_lambda   <- tf1$placeholder(dtype = "float32", shape = shape(), name = "slab_lambda")
    mixing_lambda <- tf1$placeholder(dtype = "float32", shape = shape(), name = "mixing_lambda")

    # info
    v_trans   <- params$cov_map$v_trans
    v_itrans  <- params$cov_map$v_itrans
    v_names   <- params$cov_map$v_names
    S         <- tf$constant(params$S_data[v_trans, v_trans], dtype = "float32")

    # Parameter vector
    dlt_init  <- tf$constant(params$delta_start, dtype = "float32")
    dlt_free  <- tf$constant(params$delta_free,  dtype = "float32")
    dlt_value <- tf$constant(params$delta_value, dtype = "float32")
    dlt_vec   <- tf$Variable(initial_value =          dlt_init * dlt_free + dlt_value,
                             constraint    = function(dlt) dlt * dlt_free + dlt_value, dtype = "float32")

    vec_sizes <- tf$constant(vapply(params$idx, length, 1L), dtype = "int32")
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
      psi_dup <- tf$constant(lavaan::lav_matrix_duplication(params$mat_size$psi[1]), dtype = "float32")
      psi_cc  <- tf$matmul(psi_dup, tf$expand_dims(psi_vec, 1L))
      Psi     <- tf$reshape(psi_cc, shape(params$mat_size$psi[1], params$mat_size$psi[2]))
    } else {
      Psi     <- tf$reshape(psi_vec, shape(params$mat_size$psi[1], params$mat_size$psi[2]))
    }

    # Beta matrix
    if (params$mat_size$beta[1] < 2) {
      B_0     <- tf$reshape(b_0_vec, shape(params$mat_size$beta[1], params$mat_size$beta[2]))
    } else {
      b_0_com <- tf$constant(lavaan::lav_matrix_commutation(params$mat_size$beta[1], params$mat_size$beta[2]),
                             dtype = "float32")
      B_0     <- tf$reshape(tf$matmul(b_0_com, tf$expand_dims(b_0_vec, 1L)),
                            shape(params$mat_size$beta[1], params$mat_size$beta[2]))
    }

    # Lambda matrix
    if (params$mat_size$lambda[2] < 2) {
      Lambda  <- tf$reshape(lam_vec, shape(params$mat_size$lambda[1], params$mat_size$lambda[2]))
    } else {
      lam_com <- tf$constant(lavaan::lav_matrix_commutation(params$mat_size$lambda[1], params$mat_size$lambda[2]),
                             dtype = "float32")
      Lambda  <- tf$reshape(tf$matmul(lam_com, tf$expand_dims(lam_vec, 1L)),
                            shape(params$mat_size$lambda[1], params$mat_size$lambda[2]))
    }

    # Theta matrix
    tht_dup   <- tf$constant(lavaan::lav_matrix_duplication(params$mat_size$theta[1]), dtype = "float32")
    tht_cc    <- tf$matmul(tht_dup, tf$expand_dims(tht_vec, 1L))

    Theta     <- tf$reshape(tht_cc, shape(params$mat_size$theta[1],
                                          params$mat_size$theta[2]))

    # loss function
    I_mat     <- tf$eye(params$mat_size$beta[1], dtype = "float32")
    B         <- I_mat - B_0
    B_inv     <- tf$linalg$inv(B)
    Sigma     <- tf$matmul(tf$matmul(Lambda, tf$matmul(tf$matmul(B_inv, Psi), B_inv, transpose_b = TRUE)),
                           Lambda, transpose_b = TRUE) + Theta
    Sigma_inv <- tf$linalg$inv(Sigma)

    # penalties
    one <- tf$constant(1.0, dtype = "float32")
    penalty <-
      lasso_beta   * tf$reduce_sum(tf$abs(B_0)) +
      lasso_lambda * tf$reduce_sum(tf$abs(Lambda)) +
      lasso_psi    * tf$reduce_sum(tf$abs(Psi)) +
      ridge_beta   * tf$reduce_sum(tf$square(B_0)) +
      ridge_lambda * tf$reduce_sum(tf$square(Lambda)) +
      ridge_psi    * tf$reduce_sum(tf$square(Psi)) +
      mixing_lambda         * spike_lambda * tf$reduce_sum(tf$abs(Lambda)) +
      (one - mixing_lambda) * slab_lambda  * tf$reduce_sum(tf$square(Lambda))

    # fit function
    fit <- switch(params$fit_fun,
      ml  = tf$linalg$logdet(Sigma) + tf$linalg$trace(tf$matmul(S, Sigma_inv)),
      lad = tf$reduce_sum(tf$abs(Sigma - S))
    )


    loss      <- fit + penalty

    # loglik term
    logdetS   <- tf$linalg$logdet(S)

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


    # optim
    optim <- tf1$train$AdamOptimizer()
    train <- optim$minimize(loss)

    # create configuration protobuf turning off memory optimization
    # https://github.com/tensorflow/tensorflow/issues/23780
    config_pb <- reticulate::py_run_string("
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
session_config = tf.compat.v1.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
#session_config.graph_options.rewrite_options.arithmetic_optimization = off
session_config.graph_options.rewrite_options.memory_optimization = off
    ")$session_config
    session <- tf1$Session()#config = config_pb)
    session$run(tf1$global_variables_initializer())

  })

  return(tf_env)
}
