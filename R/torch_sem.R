#' Structural equation model with a Torch backend
#'
#' Function for creating a structural equation model
#'
#' @param syntax lavaan syntax for the SEM model
#' @param dtype (optional) torch dtype for the model (default torch_float32())
#' @param device (optional) device type to put the model on. see [torch::torch_device()]
#'
#' @return A `torch_sem` object, which is an `nn_module` (torch object)
#'
#' @details
#' This function instantiates a torch object for computing the model-implied covariance matrix
#' based on a structural equation model. Through `torch`, gradients of this forward model can then
#' be computed using backpropagation, and the parameters can be optimized using gradient-based
#' optimization routines from the `torch` package.
#'
#' Because of this, it is easy to add additional penalties to the standard objective function,
#' or to write a new objective function altogether.
#'
#' @import torch
#' @import lavaan
#' @importFrom R6 R6Class
#'
#' @name torch_sem
#'
#' @export
torch_sem <- torch::nn_module(
  classname = "torch_sem",

  #' @section Methods:
  #'
  #' ## `$initialize()`
  #' The initialize method. Don't use this, just use [torch_sem()]
  #'
  #' ### Arguments
  #' - `syntax` lavaan syntax for the SEM model
  #' - `dtype` (optional) torch dtype for the model (default torch_float32())
  #' - `device` (optional) device type to put the model on. see [torch::torch_device()]
  #'
  #' ### Value
  #' A `torch_sem` object, which is an `nn_module` (torch object)
  initialize = function(syntax, dtype = torch_float32(), device = torch_device("cpu")) {
    # store params
    self$syntax <- syntax
    self$device <- device
    self$dtype <- dtype

    # compute torch settings
    self$opt <- syntax_to_torch_opts(syntax)

    # initialize the parameter vector
    self$dlt_vec <- nn_parameter(torch_tensor(self$opt$delta_start, dtype = dtype, requires_grad = TRUE, device = device))

    # initialize the parameter constraints
    self$dlt_free <- torch_tensor(self$opt$delta_free, dtype = torch_bool(), requires_grad = FALSE, device = device)
    self$dlt_value <- torch_tensor(self$opt$delta_value, dtype = dtype, requires_grad = FALSE, device = device)

    # duplication indices transforming vech to vec for psi and theta
    self$psi_dup_idx <- torch_tensor(vech_dup_idx(self$opt$psi_shape[1]), dtype = torch_long(), requires_grad = FALSE, device = device)
    self$tht_dup_idx <- torch_tensor(vech_dup_idx(self$opt$tht_shape[1]), dtype = torch_long(), requires_grad = FALSE, device = device)

    # tensor identity matrix
    self$I_mat <- torch_eye(self$opt$b_0_shape[1], dtype = dtype, requires_grad = FALSE, device = device)

    # mean is fixed to 0
    self$mu <- torch_zeros(self$opt$tht_shape[1], dtype = self$dtype, requires_grad = FALSE, device = device)
  },

  #' @section Methods:
  #'
  #' ## `$forward()`
  #' Compute the model-implied covariance matrix.
  #' Don't use this; `nn_modules` are callable, so access this method by calling
  #' the object itself as a function, e.g., `my_torch_sem()`.
  #' In the forward pass, we apply constraints to the parameter vector, and we
  #' create matrix views from it to compute the model-implied covariance matrix.
  #'
  #' ### Value
  #' A `torch_tensor` of the model-implied covariance matrix
  forward = function() {
    # apply constraints
    self$dlt <- torch_where(self$dlt_free, self$dlt_vec, self$dlt_value)

    # create the model matrix views of the delta vector
    self$dlt_split <- torch_split(self$dlt, self$opt$delta_sizes)

    # create individual matrix views
    self$Lam <- self$dlt_split[[1]]$view(c(self$opt$lam_shape[2], self$opt$lam_shape[1]))$t()
    self$Tht <- torch_index_select(self$dlt_split[[2]], 1, self$tht_dup_idx)$view(self$opt$tht_shape)
    self$Psi <- torch_index_select(self$dlt_split[[3]], 1, self$psi_dup_idx)$view(self$opt$psi_shape)
    self$B_0 <- self$dlt_split[[4]]$view(c(self$opt$b_0_shape[2], self$opt$b_0_shape[1]))$t()

    # Now create model-implied covariance matrix
    self$B <- self$I_mat - self$B_0
    self$B_inv <- torch_inverse(self$B)
    self$Sigma <-
      self$Lam$mm(self$B_inv$mm(self$Psi)$mm(self$B_inv$t()))$mm(self$Lam$t())$add(self$Tht)

    return(self$Sigma)
  },

  #' @section Methods:
  #'
  #' ## `$inverse_Hessian(loss)`
  #' Compute and return the asymptotic covariance matrix of the parameters with
  #' respect to the loss function
  #'
  #' ### Arguments
  #' - `loss` torch_tensor of freshly computed loss function (needed by torch
  #' for backwards pass)
  #'
  #' ### Value
  #' A `torch_tensor`, representing the ACOV of the free parameters
  inverse_Hessian = function(loss) {
    g <- autograd_grad(loss, self$dlt_vec, create_graph = TRUE)[[1]]
    H <- torch_jacobian(g, self$dlt_vec)
    free_idx <- torch_nonzero(self$dlt_free)$view(-1)
    self$Hinv <- torch_inverse(H[free_idx, ][, free_idx])
    return(self$Hinv)
  },

  #' @section Methods:
  #'
  #' ## `$standard_errors(loss)`
  #' Compute and return observed information standard errors of the
  #' parameters, assuming the loss function is the likelihood and the
  #' current estimates are ML estimates.
  #'
  #' ### Arguments
  #' - `loss` torch_tensor of freshly computed loss function (needed by torch
  #' for backwards pass)
  #'
  #' ### Value
  #' A `numeric vector` of standard errors of the free parameters
  standard_errors = function(loss) {
    hess <- self$inverse_Hessian(loss)
    return(as_array(hess$diag()$sqrt()))
  },

  #' @section Methods:
  #'
  #' ## `$partable(loss)`
  #'
  #' Create a lavaan-like parameter table from the current parameter estimates in the
  #' torch_sem object.
  #'
  #' ### Arguments
  #' - `loss` (optional) torch_tensor of freshly computed loss function (needed by torch
  #' for backwards pass)
  #'
  #' ### Value
  #' lavaan partable object
  partable = function(loss) {
    fit <- lavaan::sem(self$syntax, std.lv = TRUE, information = "observed",
                       fixed.x = FALSE, do.fit = FALSE)
    pt <- lavaan::partable(fit)
    idx <- which(pt$free != 0)[unique(unlist(fit@Model@x.free.idx))]
    pt[idx,"est"] <- self$free_params
    if (!missing(loss)) pt[idx,"se"] <- self$standard_errors(loss)
    return(pt)
  },

  #' @section Methods:
  #'
  #' ## `$fit(dat, lrate, maxit, verbose, tol)`
  #' Fit a torch_sem model using the default maximum likelihood objective.
  #' This function uses the Adam optimizer to estimate the parameters of a torch_sem
  #'
  #' ### Arguments
  #' - `dat` dataset (centered!) as a `torch_tensor`
  #' - `lrate` learning rate of the Adam optimizer.
  #' - `maxit` maximum number of epochs to train the model
  #' - `verbose` whether to print progress to the console
  #' - `tol` parameter change tolerance for stopping training
  #'
  #' ### Value
  #' Self, i.e., the `torch_sem` object with updated parameters
  fit = function(dat, lrate = 0.01, maxit = 5000, verbose = TRUE, tol = 1e-20) {
    if (verbose) cat("Fitting SEM with Adam optimizer and MVN log-likelihood loss\n")
    optim <- optim_adam(self$parameters, lr = lrate)
    prev_loss <- 0.0
    for (epoch in 1:maxit) {
      optim$zero_grad()
      loss <- -self$loglik(dat)
      if (verbose) {
        cat("\rEpoch:", epoch, " loglik:", -loss$item())
        flush.console()
      }
      loss$backward()
      optim$step()
      if (epoch > 1 && abs(loss$item() - prev_loss) < tol) {
        if (verbose) cat("\n")
        break
      }
      prev_loss <- loss$item()
    }
    if (epoch == maxit) warning("maximum iterations reached")
    return(invisible(self))
  },

  #' @section Methods:
  #'
  #' ## `$loglik(dat)`
  #' Multivariate normal log-likelihood of the data.
  #'
  #' ### Arguments
  #' - `dat` dataset (centered!) as a `torch_tensor`
  #'
  #' ### Value
  #' Log-likelihood value (torch scalar)
  loglik = function(dat) {
    px <- distr_multivariate_normal(loc = self$mu, covariance_matrix = self$forward())
    px$log_prob(dat)$sum()
  },

  active = list(
    #' @field free_params Vector of free parameters
    free_params = function() {
      out <- self$dlt_vec[torch_nonzero(self$dlt_free)$view(-1)]
      if (self$device$type != "cpu") return(as_array(out$cpu()))
      return(as_array(out))
    }
  )
)
