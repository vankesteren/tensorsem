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
  pt <- lavaan::lavaanify(
    mod,
    model.type      = "sem",
    auto            = TRUE,
    int.ov.free     = TRUE,
    int.lv.free     = FALSE,
    auto.fix.first  = FALSE,
    auto.fix.single = TRUE,
    auto.var        = TRUE,
    auto.cov.lv.x   = TRUE,
    auto.cov.y      = TRUE,
    auto.th         = TRUE,
    auto.delta      = TRUE,
    conditional.x   = FALSE,
    fixed.x         = FALSE,
    std.lv          = TRUE
  )

  # lav options corresponding to sem()
  lo <- lavaan::lavOptions()
  lo$missing         <- "ml"
  lo$int.ov.free     <- TRUE
  lo$int.lv.free     <- FALSE
  lo$auto.fix.first  <- TRUE
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
  mis_idx  <- which(is.na(sub_dat))
  mis_mat  <- matrix(1, nrow(sub_dat), ncol(sub_dat))
  mis_mat[mis_idx] <- 0

  # trans stuff
  v_itrans  <- vapply(colnames(sub_dat), function(var) which(var == lav_mod@dimNames[[2]][[1]]), 1L)
  v_trans   <- vapply(lav_mod@dimNames[[2]][[1]], function(var) which(var == colnames(sub_dat)), 1L)

  # get starting values / set values
  # S_data   <- cov(sub_dat, use = "pairwise") * (nrow(sub_dat) - 1) / nrow(sub_dat)
  s_stats  <- getFromNamespace("lav_samplestats_from_data", "lavaan")(
    lavdata       =  getFromNamespace("lavData", "lavaan")(data = data, lavoptions = lo),
    missing       = lo$missing,
    rescale       =
      (lo$estimator %in% c("ML","REML","NTRLS") &&
         lo$likelihood == "normal"),
    estimator     = lo$estimator,
    mimic         = lo$mimic,
    meanstructure = TRUE,
    conditional.x = lo$conditional.x,
    fixed.x       = lo$fixed.x,
    group.w.free  = lo$group.w.free,
    missing.h1    = (lo$missing != "listwise"),
    WLS.V             = NULL,
    NACOV             = NULL,
    gamma.n.minus.one = lo$gamma.n.minus.one,
    se                = lo$se,
    information       = lo$information,
    ridge             = lo$ridge,
    optim.method      = lo$optim.method.cor,
    zero.add          = lo$zero.add,
    zero.keep.margins = lo$zero.keep.margins,
    zero.cell.warn    = lo$zero.cell.warn
  )
  pt$start <- getFromNamespace("lav_start", "lavaan")(lavpartable = pt, lavsamplestats = s_stats, model.type = "sem")

  lav_mod  <- getFromNamespace("lav_model", "lavaan")(lavpartable = pt, lavoptions  = lo)

  # create vectors
  psi_vec <- lavaan::lav_matrix_vech(lav_mod@GLIST$psi)
  if (is.null(lav_mod@GLIST$beta)) {
    b_0_vec <- rep(0, prod(dim(lav_mod@GLIST$psi)))
  } else {
    b_0_vec <- lavaan::lav_matrix_vec(lav_mod@GLIST$beta)
  }
  lam_vec <- lavaan::lav_matrix_vec(lav_mod@GLIST$lambda)
  tht_vec <- lavaan::lav_matrix_vech(lav_mod@GLIST$theta)

  # 1 - 0 free param matrices
  glist_free <- list()
  for (mm in seq_along(lav_mod@GLIST)) {
    mat <- lav_mod@GLIST[[mm]]
    mat[1:length(mat)] <- 0
    free_idx <- lav_mod@m.free.idx[[mm]]
    if (length(free_idx) > 0) mat[free_idx] <- 1
    glist_free[[names(lav_mod@GLIST)[[mm]]]] <- mat
  }

  psi_free <- lavaan::lav_matrix_vech(glist_free$psi)
  if (is.null(glist_free$beta)) {
    b_0_free <- rep(0, prod(dim(lav_mod@GLIST$psi)))
  } else {
    b_0_free <- lavaan::lav_matrix_vec(glist_free$beta)
  }
  lam_free <- lavaan::lav_matrix_vec(glist_free$lambda)
  tht_free <- lavaan::lav_matrix_vech(glist_free$theta)


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
  if (is.null(mat_siz$beta)) mat_siz$beta <- mat_siz$psi

  return(list(
    mat_size     = mat_siz,
    fit_fun      = "ml",
    delta_start  = delta_start,
    delta_free   = delta_free,
    delta_value  = delta_value,
    data_mat     = scale(as.matrix(sub_dat)[, v_trans], scale = FALSE),
    miss_mat     = mis_mat[,v_trans],
    polyak_decay = 0.98,
    idx          = list(
      psi = psi_idx,
      b_0 = b_0_idx,
      lam = lam_idx,
      tht = tht_idx
    ),
    cov_map      = list(
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

    psi_dup <- lavaan::lav_matrix_duplication(mat_size$psi[1])
    tht_dup <- lavaan::lav_matrix_duplication(mat_size$theta[1])

    list(
      psi    = matrix(c(psi_dup %*% psi_vec), mat_size$psi[1], mat_size$psi[2]),
      beta   = matrix(b_0_vec, mat_size$beta[1], mat_size$beta[2], byrow = TRUE),
      lambda = matrix(lam_vec, mat_size$lambda[1], mat_size$lambda[2]),
      theta  = matrix(c(tht_dup %*% tht_vec), mat_size$theta[1], mat_size$theta[2])
    )
  })
}
