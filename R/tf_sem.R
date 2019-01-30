#' Perform structural equation modeling with TensorFlow
#'
#' This function converts lavaan code and a dataset to a TensorFlow object. This
#' object can then be used to train the model and view estimates.
#'
#' @param lav_model A lavaan syntax model
#' @param data A data frame. Only numeric variables supported.
#' @param fit (optional) train the model for 2000 iterations upon creation
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
#'      Plots the loss curve with iteration number on the x-axis and the
#'      value of the loss function on the y-axis.
#'   }
#' }
#'
#' @export
tf_sem <- function(lav_model, data, fit = FALSE) {

  tf_params     <- lav_to_tf_pars(lav_model, data)
  tf_session    <- tf_pars_to_session(tf_params)
  tf_sem_object <- tf_sem_object$new(tf_session, lav_model, nrow(data))

  if (fit) tf_sem_object$train(2000)

  return(tf_sem_object)
}
