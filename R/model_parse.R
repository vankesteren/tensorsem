#' Create a torch options list from lavaan syntax.
#'
#' @param syntax lavaan syntax
#'
#' @return list of tensorsem options
#'
#' @export
syntax_to_torch_opts <- function(syntax) {
  # create lavaan model
  fit <- lavaan::sem(syntax, std.lv = TRUE, information = "observed",
                     fixed.x = FALSE, do.fit = FALSE)
  return(lav_mod_to_torch_opts(fit@Model))
}

#' Create a torch options list from a lavaan Model class.
#'
#' @param lav_mod lavaan Model class object
#'
#' @return list of tensorsem options
#'
#' @export
lav_mod_to_torch_opts <- function(lav_mod) {
  # create matrix representation
  # parameter estimates
  glist <- lav_mod@GLIST

  # 1 - 0 free param matrices
  free_idx <- lav_mod@m.free.idx
  glist_free <- list()
  for (mm in seq_along(glist)) {
    mat <- glist[[mm]]
    mat[1:length(mat)] <- 0
    if (length(free_idx[[mm]]) > 0) mat[free_idx[[mm]]] <- 1
    glist_free[[names(glist)[[mm]]]] <- mat
  }


  # create vectors
  lam_vec  <- lavaan::lav_matrix_vec(glist$lambda)
  lam_free <- lavaan::lav_matrix_vec(glist_free$lambda)

  tht_vec  <- lavaan::lav_matrix_vech(glist$theta)
  tht_free <- lavaan::lav_matrix_vech(glist_free$theta)

  psi_vec  <- lavaan::lav_matrix_vech(glist$psi)
  psi_free <- lavaan::lav_matrix_vech(glist_free$psi)

  if (is.null(glist$beta)) {
    b_0_vec <- b_0_free <- rep(0, prod(dim(glist$psi)))
  } else {
    b_0_vec  <- lavaan::lav_matrix_vec(glist$beta)
    b_0_free <- lavaan::lav_matrix_vec(glist_free$beta)
  }

  # actual params vectors
  delta_start <- c(lam_vec, tht_vec, psi_vec, b_0_vec)
  delta_free  <- c(lam_free, tht_free, psi_free, b_0_free)
  delta_value <- delta_start * (1 - delta_free)
  shapes <- lapply(lav_mod@GLIST, dim)
  if (is.null(shapes$beta)) shapes$beta <- shapes$psi

  return(list(
    # actual params
    delta_start = delta_start,
    delta_free  = delta_free,
    delta_value = delta_value,
    delta_sizes = sapply(list(lam_vec, tht_vec, psi_vec, b_0_vec), length),
    psi_shape   = shapes$psi,
    b_0_shape   = shapes$beta,
    lam_shape   = shapes$lambda,
    tht_shape   = shapes$theta,
    ov_names    = lav_mod@dimNames[[2]][[1]]
  ))
}

#' Create a lavaan parameter table from torch free_params output
#'
#' See examples in tensorsem for how to save the output.
#'
#' @param pars data frame of parameter estimates (est) and their standard errors (se)
#' @param syntax syntax of the original model
#'
#' @export
partable_from_torch <- function(pars, syntax) {
  fit <- lavaan::sem(syntax, std.lv = TRUE, information = "observed",
                     fixed.x = FALSE, do.fit = FALSE)
  pt <- lavaan::partable(fit)
  idx <- which(pt$free != 0)[unique(unlist(fit@Model@x.free.idx))]
  pt[idx,"est"] <- pars$est
  pt[idx,"se"] <- pars$se
  return(pt)
}
