library(torch)
library(lavaan)

torch_sem <- nn_module(
  classname = "torch_sem",

  initialize = function(model, dtype = torch_float32()) {
    # In The constructor we instantiate the parameter vector, its constraints,
    # as well as several helper lists and tensors for creating the model from
    # this parameter vector
    # :param opt_dict: dictionary from lav_to_tf_pars
    # :param dtype: data type, default float32

    self$opt <- syntax_to_torch_opts(model)

    # initialize the parameter vector
    self$dlt_vec <- nn_parameter(torch_tensor(self$opt$delta_start, dtype = dtype, requires_grad = TRUE))

    # initialize the parameter constraints
    self$dlt_free <- torch_tensor(self$opt$delta_free, dtype = torch_bool(), requires_grad = FALSE)
    self$dlt_value <- torch_tensor(self$opt$delta_value, dtype = dtype, requires_grad = FALSE)

    # duplication indices transforming vech to vec for psi and theta
    self$psi_dup_idx <- torch_tensor(dup_idx(self$opt$psi_shape[1]), dtype = torch_long(), requires_grad = FALSE)
    self$tht_dup_idx <- torch_tensor(dup_idx(self$opt$tht_shape[1]), dtype = torch_long(), requires_grad = FALSE)

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
    self$Lam <- self$dlt_split[[1]]$view(c(self$opt$lam_shape[2], self$opt$lam_shape[1])) |> torch_t()
    self$Tht <- torch_index_select(self$dlt_split[[2]], 1, self$tht_dup_idx)$view(self$opt$tht_shape)
    self$Psi <- torch_index_select(self$dlt_split[[3]], 1, self$psi_dup_idx)$view(self$opt$psi_shape)
    self$B_0 <- self$dlt_split[[4]]$view(c(self$opt$b_0_shape[2], self$opt$b_0_shape[1])) |> torch_t()

    # Now create model-implied covariance matrix
    self$B <- self$I_mat - self$B_0
    self$B_inv <- torch_inverse(self$B)
    self$Sigma <-
      self$Lam |>
      torch_mm(self$B_inv |> torch_mm(self$Psi) |> torch_mm(torch_t(self$B_inv))) |>
      torch_mm(torch_t(self$Lam)) |>
      torch_add(self$Tht)

    return(self$Sigma)
  },

  Inverse_Hessian = function(loss) {
    # Computes and returns the asymptotic covariance matrix of the parameters with
    # respect to the loss function, to compute standard errors (sqrt(diag(ACOV)))
    # :param loss: freshly computed loss function (for backwards pass)
    # :return: ACOV tensor of the free parameters
    g <- autograd_grad(loss, self$dlt_vec, create_graph = TRUE)[[1]]
    H <- jacobian(g, self$dlt_vec)
    free_idx <- torch_nonzero(self$dlt_free)$view(-1)
    self$Hinv <- torch_inverse(H[free_idx, ][, free_idx])
    return(self$Hinv)
  },

  active = list(
    free_params = function(value) {
      # Get or set the free parameter vector
      # :return: Tensor with free parameters
      if (!missing(value)) stop("setting free parameters not yet supported")
      return(self$dlt_vec[torch_nonzero(self$dlt_free)$view(-1)])
    }
  )

)
