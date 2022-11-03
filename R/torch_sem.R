library(torch)
library(lavaan)

torch_sem <- nn_module(
  classname = "torch_sem",

  initialize = function(syntax, dtype = torch_float32()) {
    # In The constructor we instantiate the parameter vector, its constraints,
    # as well as several helper lists and tensors for creating the model from
    # this parameter vector
    # :param opt_dict: dictionary from lav_to_tf_pars
    # :param dtype: data type, default float32

    self$syntax <- syntax

    self$opt <- syntax_to_torch_opts(syntax)

    # initialize the parameter vector
    self$dlt_vec <- nn_parameter(torch_tensor(self$opt$delta_start, dtype = dtype, requires_grad = TRUE))

    # initialize the parameter constraints
    self$dlt_free <- torch_tensor(self$opt$delta_free, dtype = torch_bool(), requires_grad = FALSE)
    self$dlt_value <- torch_tensor(self$opt$delta_value, dtype = dtype, requires_grad = FALSE)

    # duplication indices transforming vech to vec for psi and theta
    self$psi_dup_idx <- torch_tensor(vech_dup_idx(self$opt$psi_shape[1]), dtype = torch_long(), requires_grad = FALSE)
    self$tht_dup_idx <- torch_tensor(vech_dup_idx(self$opt$tht_shape[1]), dtype = torch_long(), requires_grad = FALSE)

    # tensor identity matrix
    self$I_mat <- torch_eye(self$opt$b_0_shape[1], dtype = dtype, requires_grad = FALSE)
  },

  forward = function(input) {
    # In the forward pass, we apply constraints to the parameter vector, and we
    # create matrix views from it to compute the model-implied covariance matrix
    # :return: model-implied covariance matrix tensor

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

  inverse_Hessian = function(loss) {
    # Computes and returns the asymptotic covariance matrix of the parameters with
    # respect to the loss function, to compute standard errors (sqrt(diag(ACOV)))
    # :param loss: freshly computed loss function (for backwards pass)
    # :return: ACOV tensor of the free parameters
    g <- autograd_grad(loss, self$dlt_vec, create_graph = TRUE)[[1]]
    H <- torch_jacobian(g, self$dlt_vec)
    free_idx <- torch_nonzero(self$dlt_free)$view(-1)
    self$Hinv <- torch_inverse(H[free_idx, ][, free_idx])
    return(self$Hinv)
  },

  standard_errors = function(loss) {
    hess <- self$inverse_Hessian(loss)
    return(hess$diag()$sqrt() |> as_array())
  },

  partable = function(loss) {
    fit <- lavaan::sem(self$syntax, std.lv = TRUE, information = "observed",
                       fixed.x = FALSE, do.fit = FALSE)
    pt <- lavaan::partable(fit)
    idx <- which(pt$free != 0)[unique(unlist(fit@Model@x.free.idx))]
    pt[idx,"est"] <- self$free_params
    pt[idx,"se"] <- self$standard_errors(loss)
    return(pt)
  },

  fit = function(dat, lrate = 0.01, maxit = 5000, verbose = TRUE, tol = 1e-20) {
    cat("Fitting SEM with Adam optimizer and MVN log-likelihood loss\n")
    optim <- optim_adam(self$parameters, lr = lrate)
    prev_loss <- 0.0
    for (epoch in 1:maxit) {
      optim$zero_grad()
      loss <- mvn_negloglik(dat, tsem())
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
  },

  active = list(
    free_params = function(value) {
      # Get or set the free parameter vector
      # :return: Tensor with free parameters
      if (!missing(value)) stop("setting free parameters not yet supported")
      return(as_array(self$dlt_vec[torch_nonzero(self$dlt_free)$view(-1)]))
    }
  )

)
