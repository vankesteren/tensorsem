#' Convert a lavaan model to sem_tf parameters
#'
#' @param mod lavaan model syntax
#' @param data data frame
#'
#' @importFrom stats cov
#' @importFrom methods new
#' @importFrom utils getFromNamespace
#'
#' @keywords internal
lav_to_tf_pars <- function(mod, data) {
  # create lavaan model
  pt <- lavaan::lavaanify(mod, model.type = "sem", auto = TRUE)

  # lav options corresponding to sem()
  lo <- lavaan::lavOptions()
  lo$int.ov.free     <- TRUE
  lo$int.lv.free     <- FALSE
  lo$auto.fix.first  <- FALSE
  lo$auto.fix.single <- TRUE
  lo$auto.var        <- TRUE
  lo$auto.cov.lv.x   <- TRUE
  lo$auto.cov.y      <- TRUE
  lo$auto.th         <- TRUE
  lo$auto.delta      <- TRUE
  lo$conditional.x   <- TRUE
  lo$fixed.x         <- FALSE
  lo$representation  <- "LISREL"
  lo$std.lv          <- TRUE

  lav_mod <- getFromNamespace("lav_model", "lavaan")(lavpartable = pt, lavoptions  = lo)

  v_names  <- lav_mod@dimNames[[2]][[1]]

  sub_dat  <- data[, colnames(data) %in% v_names]

  # trans stuff
  v_itrans  <- vapply(colnames(sub_dat), function(var) which(var == lav_mod@dimNames[[2]][[1]]), 1L)
  v_trans   <- vapply(lav_mod@dimNames[[2]][[1]], function(var) which(var == colnames(sub_dat)), 1L)

  S_data   <- cov(sub_dat) * (nrow(sub_dat) - 1) / nrow(sub_dat)
  s_stats  <- new("lavSampleStats",
                  cov = list(S_data[v_trans, v_trans]),
                  mean = list(colMeans(sub_dat)[v_trans]),
                  missing.flag = FALSE)

  # get starting values / set values
  pt$start <- getFromNamespace("lav_start", "lavaan")(lavpartable = pt, lavsamplestats = s_stats, model.type = "sem")

  lav_mod  <- getFromNamespace("lav_model", "lavaan")(lavpartable = pt, lavoptions  = lo)

  # create vectors
  psi_vec <- matrixcalc::vech(lav_mod@GLIST$psi)
  if (is.null(lav_mod@GLIST$beta)) {
    b_0_vec <- 0
  } else {
    b_0_vec <- matrixcalc::vec(lav_mod@GLIST$beta)
  }
  lam_vec <- matrixcalc::vec(lav_mod@GLIST$lambda)
  tht_vec <- matrixcalc::vech(lav_mod@GLIST$theta)

  # 1 - 0 free param matrices
  glist_free <- list()
  for (mm in seq_along(lav_mod@GLIST)) {
    mat <- lav_mod@GLIST[[mm]]
    mat[1:length(mat)] <- 0
    free_idx <- lav_mod@m.free.idx[[mm]]
    if (length(free_idx) > 0) mat[free_idx] <- 1
    glist_free[[names(lav_mod@GLIST)[[mm]]]] <- mat
  }

  psi_free <- matrixcalc::vech(glist_free$psi)
  if (is.null(glist_free$beta)) {
    b_0_free <- 0
  } else {
    b_0_free <- matrixcalc::vec(glist_free$beta)
  }
  lam_free <- matrixcalc::vec(glist_free$lambda)
  tht_free <- matrixcalc::vech(glist_free$theta)


  # actual params
  delta_start <- c(psi_vec, b_0_vec, lam_vec, tht_vec)
  delta_free  <- c(psi_free, b_0_free, lam_free, tht_free)
  delta_value <- delta_start * (1 - delta_free)

  # indices
  psi_idx <- 1:length(psi_vec)
  b_0_idx <- (max(psi_idx) + 1):(max(psi_idx) + length(b_0_vec))
  lam_idx <- (max(b_0_idx) + 1):(max(b_0_idx) + length(lam_vec))
  tht_idx <- (max(lam_idx) + 1):(max(lam_idx) + length(tht_vec))

  # matrix sizes
  mat_siz <- lapply(lav_mod@GLIST, dim)
  if (is.null(mat_siz$beta)) mat_siz$beta <- c(1L, 1L)

  return(list(
    mat_size    = mat_siz,
    idx         = list(
      psi = psi_idx,
      b_0 = b_0_idx,
      lam = lam_idx,
      tht = tht_idx
    ),
    delta_start = delta_start,
    delta_free  = delta_free,
    delta_value = delta_value,
    S_data      = S_data,
    cov_map     = list(
      v_trans  = v_trans,
      v_itrans = v_itrans,
      v_names  = v_names
    )
  ))
}

#' Convert tf params to lavaan GLIST
#'
#' @keywords internal
tf_pars_to_glist <- function(tf_pars, type = "start") {
  with(tf_pars, {
    delta <- switch(type,
                    free  = delta_free,
                    value = delta_value,
                    start = delta_start
    )

    psi_vec <- delta[tf_pars$idx$psi]
    b_0_vec <- delta[tf_pars$idx$b_0]
    lam_vec <- delta[tf_pars$idx$lam]
    tht_vec <- delta[tf_pars$idx$tht]

    psi_dup <- matrixcalc::duplication.matrix(mat_size$psi[1])
    tht_dup <- matrixcalc::duplication.matrix(mat_size$theta[1])

    list(
      psi    = matrix(c(psi_dup %*% psi_vec), mat_size$psi[1], mat_size$psi[2]),
      beta   = matrix(b_0_vec, mat_size$beta[1], mat_size$beta[2], byrow = TRUE),
      lambda = matrix(lam_vec, mat_size$lambda[1], mat_size$lambda[2]),
      theta  = matrix(c(tht_dup %*% tht_vec), mat_size$theta[1], mat_size$theta[2])
    )
  })
}
