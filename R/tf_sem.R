#' Perform structural equation modeling with TensorFlow
#'
#' This function converts lavaan code and a dataset to a TensorFlow object. This
#' object can then be used to train the model and view estimates.
#'
#' @param lav_model A lavaan syntax model. See details for restrictions.
#' @param data A data frame. Only numeric variables supported.
#' @param fit (optional) train the model for 2000 iterations upon creation
#' @param loss expression
#' @param ... any hyperparameters used in the loss function
#'
#' @details The tf_sem function supports only a subset of the lavaan syntax as of now:
#' \itemize{
#'   \item{Only datasets with numeric values and no missing data}
#'   \item{Only regression ("~"), factor loadings ("=~") and variances ("~~")}
#'   \item{No multigroup SEM}
#'   \item{No intercepts / means}
#'   \item{No equality / inequality constraints}
#' }
#'
#'
#' @return a tf_sem object. See below.
#'
#' @section Using the \code{tf_sem} object:
#' See below the available methods and extractable information from the resulting object
#' \describe{
#'   \item{\code{tf_mod$train(niter = 10000, pb = TRUE, verbose = FALSE)}}{
#'     \describe{
#'       \item{Description}{
#'         This function runs the Adam optimizer on the tensorflow session, and
#'         writes the values of the loss function to this object for inspection.
#'         Make sure to perform enough iterations (i.e., loss should not change)
#'         before inspecting the results.
#'       }
#'       \item{Arguments}{
#'         \itemize{
#'           \item{\code{niter}}{ How many iterations to run}
#'           \item{\code{pb}}{ Whether to display a progress bar}
#'           \item{\code{verbose}}{ Whether to display parameters during training}
#'         }
#'       }
#'     }
#'   }
#'   \item{\code{tf_mod$print()}}{Prints basic information about the model object}
#'   \item{\code{tf_mod$summary()}}{Prints summary information, including all the model matrices.}
#'   \item{\code{tf_mod$gradients()}}{Prints gradient for each element of the parameter matrices.}
#'   \item{\code{tf_mod$plot_loss()}}{
#'     Plots the loss curve with iteration number on the x-axis and the
#'     value of the loss function on the y-axis.
#'   }
#'   \item{\code{tf_mod$Sigma}}{Returns the Sigma matrix}
#'   \item{\code{tf_mod$Psi}}{Returns the Psi matrix}
#'   \item{\code{tf_mod$Beta}}{Returns the Beta matrix}
#'   \item{\code{tf_mod$Lambda}}{Returns the Lambda matrix}
#'   \item{\code{tf_mod$Theta}}{Returns the Theta matrix}
#'   \item{\code{tf_mod$Psi_grad}}{Returns the gradient of the Psi matrix}
#'   \item{\code{tf_mod$Beta_grad}}{Returns the gradient of the Beta matrix}
#'   \item{\code{tf_mod$Lambda_grad}}{Returns the gradient of the Lambda matrix}
#'   \item{\code{tf_mod$Theta_grad}}{Returns the gradient of the Theta matrix}
#'   \item{\code{tf_mod$data}}{Returns the observed covariance matrix}
#'   \item{\code{tf_mod$loss}}{Returns the current value of the loss function}
#'   \item{\code{tf_mod$loglik}}{Returns the current log-likelihood value}
#'   \item{\code{tf_mod$delta}}{Returns the full delta parameter vector}
#'   \item{\code{tf_mod$delta_idx}}{Returns the indices of the free elements of delta}
#'   \item{\code{tf_mod$delta_free}}{Returns the values of the free elements of delta}
#'   \item{\code{tf_mod$delta_grad}}{Returns the gradient of the loss w.r.t. delta}
#'   \item{\code{tf_mod$delta_hess}}{Returns the hessian of the loss w.r.t. delta}
#'   \item{\code{tf_mod$ACOV}}{
#'     Returns the asymptotic covariance matrix of the free elements of delta,
#'     assuming that the loss function is proportional to the maximum likelihood
#'     fit function.
#'   }
#' }
#'
#' @examplesn
#' \donttest{
#'   mod    <- "x1 ~ x2 + x3"
#'   dat    <- lavaan::HolzingerSwineford1939
#'   tf_mod <- tf_sem(mod, dat)
#'   tf_mod$train(50)
#'   tf_mod$summary()
#' }
#'
#' @export
tf_sem <- function(lav_model, data, fit = FALSE, loss = ml_loss, ...) {

  tf_params     <- lav_to_tf_pars(lav_model, data)
  tf_session    <- tf_pars_to_session(tf_params)
  tf_sem_object <- tf_sem_object$new(tf_session, lav_model, nrow(data))

  if (fit) tf_sem_object$train(2000)

  return(tf_sem_object)
}
