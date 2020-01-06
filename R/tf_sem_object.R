#' TensorFlow SEM object
#'
#' An R6 object with an initialized TensorFlow environment and user-friendly methods and bindings to this environment
#'
#' @importFrom R6 R6Class
#' @importFrom stringr str_pad str_trim
#' @importFrom progress progress_bar
#'
#' @keywords internal
tf_sem_object <- R6Class(
  classname = "tf_sem",
  public = list(
    tf_session    = NULL,
    lav_model     = NULL,
    sample_size   = NULL,
    loss_vec      = NULL,
    loss_vec_sgd  = NULL,
    penalties     = list(
      lasso_beta     = 0.0,
      lasso_lambda   = 0.0,
      lasso_psi      = 0.0,
      ridge_beta     = 0.0,
      ridge_lambda   = 0.0,
      ridge_psi      = 0.0,
      spike_lambda   = 0.0,
      slab_lambda    = 0.0,
      mixing_lambda  = 0.0
    ),
    feed          = NULL,
    polyak_result = FALSE,
    initialize    = function(tf_session, mod, sample_size) {
      self$tf_session  <- tf_session
      self$lav_model   <- mod
      self$sample_size <- sample_size
      private$update_feed()
      self$loss_vec    <- self$loss
    },
    finalize      = function() {
      # close tensorflow session when object is garbage collected
      self$tf_session$session$close()
    },

    # Methods
    train         = function(niter = 10000, pb = TRUE, verbose = FALSE) {
      private$update_feed()
      loss_vec     <- numeric(niter)
      loss_vec_sgd <- numeric(niter * self$sample_size)

      if (verbose) {
        cat(str_pad("ITER", nchar(niter)), " | ", str_pad("LOSS", nchar(format(round(self$loss, 5)))), " | ",
            "FIRST 10 FREE PARAMETERS\n")
      } else if (pb) {
        progbar <- progress_bar$new(format = "[:spin] [epoch: :epoch] [loss: :loss] [:bar] :percent", total = niter * self$sample_size)
      }

      for (iter in 1:niter) {
        on.exit({
          self$loss_vec <- c(self$loss_vec, loss_vec[1:iter])
          self$loss_vec_sgd <- c(self$loss_vec_sgd, loss_vec_sgd[1:(self$sample_size * (iter - 1) + n)])
        })

        private$run(self$tf_session$dat$iter$initializer)

        loss <- 0.0
        n    <- 1L
        tfdatasets::until_out_of_range({
          result <- private$run(list(self$tf_session$loss, self$tf_session$train))
          loss   <- loss + result[[1]]
          loss_vec_sgd[self$sample_size * (iter - 1) + n] <- result[[1]]
          n <- n + 1L
          if (pb && !verbose) {
            progbar$tick(tokens = list(
              epoch = iter,
              loss = format(round(ifelse(iter > 1, loss_vec[iter - 1], 0), 5), nsmall = 5))
            )
          }
        })

        loss_vec[iter] <- loss

        if (verbose && iter %% ceiling(niter / 100) == 0) {
          freeparams <- self$delta_free
          if (length(freeparams) > 10) freeparams <- freeparams[1:10]

          cat(str_pad(iter, nchar(niter)), " | ",
              format(round(loss_vec[iter], 5), nsmall = 5), " | ",
              format(round(freeparams,  3), nsmall = 3), "\n")
        }
      }

      # self$loss_vec <- c(self$loss_vec, loss_vec)

      return(invisible(self))
    },
    print         = function() {
      cat("\nTensorFlow SEM session\n----------------------\n\n")
      cat("Model:", str_trim(self$lav_model), sep = "\n")
      cat("\nIters:", length(self$loss_vec) - 1)
      return(invisible(self))
    },
    summary       = function() {
      cat("\nTensorFlow SEM session\n----------------------\n\n")
      loss <- tryCatch(self$loss, error = function(e) Inf)
      cat("Loss:", loss, "\n")
      cat("\n\nSigma:\n")
      print(self$Sigma)
      cat("\n\nPsi:\n")
      print(self$Psi)
      cat("\n\nBeta:\n")
      print(self$Beta)
      cat("\n\nLambda:\n")
      print(self$Lambda)
      cat("\n\nTheta:\n")
      print(self$Theta)

      return(invisible(self))
    },
    gradients     = function() {
      cat("\nTensorFlow SEM gradients\n------------------------\n\n")
      cat("\nPsi:\n")
      print(self$Psi_grad)
      cat("\n\nBeta:\n")
      print(self$Beta_grad)
      cat("\n\nLambda:\n")
      print(self$Lambda_grad)
      cat("\n\nTheta:\n")
      print(self$Theta_grad)

      return(invisible(self))
    },
    plot_loss     = function(.type = "Epoch", ...) {
      if (.type == "Epoch") {
        y <- self$loss_vec
      } else if (.type == "SGD") {
        y <- self$loss_vec_sgd
      } else if (.type == "Moving") {
        y <- diff(cumsum(na.omit(self$loss_vec_sgd)), lag = self$sample_size)
      }
      if (length(y) < 2) stop("Too few iterations to plot loss.")
      plot(x    = 1:length(y),
           y    = y,
           xlab = "Iterations",
           ylab = "Loss",
           main = paste(.type, "loss plot"),
           bty  = "L",
           type = "l",
           col  = "#00008b",
           ...
      )
    },
    set_init      = function(lav) {
      self$Psi    <- lav@Model@GLIST$psi
      if (!is.null(lav@Model@GLIST$beta)) {
        self$Beta   <- lav@Model@GLIST$beta
      }
      self$Lambda <- lav@Model@GLIST$lambda
      self$Theta  <- lav@Model@GLIST$theta
    }
  ),
  private = list(
    update_feed   = function() {
      # create hyperparameter feed
      feed_list        <- self$penalties
      names(feed_list) <- sapply(names(self$penalties), function(n) self$tf_session[[n]]$name)
      self$feed        <- tensorflow::dict(feed_list)
    },
    run           = function(...) {
      self$tf_session$session$run(..., feed_dict = self$feed)
    },
    input_values  = function(val_vec, target) {
      orig_vec <- private$run(self$tf_session$dlt_vec)
      vec_size <- private$run(self$tf_session$vec_sizes)
      inp_idx  <- switch(target,
        Psi    = 1:vec_size[1],
        Beta   = (vec_size[1] + 1):sum(vec_size[1:2]),
        Lambda = (sum(vec_size[1:2]) + 1):sum(vec_size[1:3]),
        Theta  = (sum(vec_size[1:3]) + 1):sum(vec_size[1:4])
      )
      orig_vec[inp_idx] <- val_vec
      self$tf_session$dlt_vec$load(orig_vec, self$tf_session$session)
    }
  ),
  private = list(
    update_feed   = function() {
      # create hyperparameter feed
      feed_list        <- self$penalties
      names(feed_list) <- sapply(names(self$penalties), function(n) self$tf_session[[n]]$name)
      self$feed        <- tensorflow::dict(feed_list)
    },
    run           = function(...) {
      self$tf_session$session$run(..., feed_dict = self$feed)
    }
  ),
  active = list(
    # matrices
    Sigma         = function() {
      if (self$polyak_result) {
        return(private$run(self$tf_session$polyak$average(self$tf_session$Sigma_ful)))
      } else {
        return(private$run(self$tf_session$Sigma_ful))
      }
    },
    Psi           = function(Psi_new) {
      if (self$polyak_result) {
        Psi <- private$run(self$tf_session$polyak$average(self$tf_session$Psi))
      } else {
        Psi <- private$run(self$tf_session$Psi)
      }
      if (missing(Psi_new)) return(Psi)
      if (!identical(dim(Psi_new), dim(Psi))) stop("Input size not equal to tensor size")
      private$input_values(c(Psi_new[lower.tri(Psi_new, diag = TRUE)]), "Psi")
    },
    Beta          = function(Beta_new) {
      if (self$polyak_result) {
        Beta <- private$run(self$tf_session$polyak$average(self$tf_session$B_0))
      } else {
        Beta <- private$run(self$tf_session$B_0)
      }
      if (missing(Beta_new)) return(Beta)
      if (!identical(dim(Beta_new), dim(Beta))) stop("Input size not equal to tensor size")
      private$input_values(c(Beta_new), "Beta")
    },
    Lambda        = function(Lambda_new) {
      if (self$polyak_result) {
        Lambda <- private$run(self$tf_session$polyak$average(self$tf_session$Lambda))
      } else {
        Lambda <- private$run(self$tf_session$Lambda)
      }
      if (missing(Lambda_new)) return(Lambda)
      if (!identical(dim(Lambda_new), dim(Lambda))) stop("Input size not equal to tensor size")
      private$input_values(c(Lambda_new), "Lambda")
    },
    Theta         = function(Theta_new) {
      if (self$polyak_result) {
        Theta <- private$run(self$tf_session$polyak$average(self$tf_session$Theta))
      } else {
        Theta <- private$run(self$tf_session$Theta)
      }
      if (missing(Theta_new)) return(Theta)
      if (!identical(dim(Theta_new), dim(Theta))) stop("Input size not equal to tensor size")
      private$input_values(c(Theta_new[lower.tri(Theta, diag = TRUE)]), "Theta")
    },

    # gradients
    Psi_grad      = function() {
      if (self$polyak_result) {
        private$run(self$tf_session$polyak$average(self$tf_session$Psi_g))
      } else {
        private$run(self$tf_session$Psi_g)
      }
    },
    Beta_grad     = function() {
      if (self$polyak_result) {
        private$run(self$tf_session$polyak$average(self$tf_session$B_0_g))
      } else {
        private$run(self$tf_session$B_0_g)
      }
    },
    Lambda_grad   = function() {
      if (self$polyak_result) {
        private$run(self$tf_session$polyak$average(self$tf_session$Lambda_g))
      } else {
        private$run(self$tf_session$Lambda_g)
      }
    },
    Theta_grad    = function() {
      if (self$polyak_result) {
        private$run(self$tf_session$polyak$average(self$tf_session$Theta_g))
      } else {
        private$run(self$tf_session$Theta_g)
      }
    },

    # data & loss
    data          = function() {
      dat <- utils::read.csv(self$tf_session$dat$data_loc)
      msk <- utils::read.csv(self$tf_session$dat$mask_loc)
      dat[msk == 0] <- NA
      colnames(dat) <- self$tf_session$v_names[self$tf_session$v_trans]
      dat
    },
    loss          = function() {
      private$run(self$tf_session$dat$iter$initializer)
      loss <- 0.0
      tfdatasets::until_out_of_range(
        loss <- loss + private$run(self$tf_session$loss)
      )
      loss
    },
    fit_value     = function() {
      private$run(self$tf_session$dat$iter$initializer)
      fit_value <- 0.0
      tfdatasets::until_out_of_range(
        fit_value <- fit_value + private$run(self$tf_session$fit)
      )
      fit_value
    },
    loglik        = function() {
      lavaan:::lav_mvnorm_missing_llik_casewise(
        Y = self$data,
        wt = NULL,
        Mu = rep(0, self$tf_session$dat$n_col),
        Sigma = self$Sigma)
    },

    # param vec
    delta         = function() {
      if (self$polyak_result) {
        return(private$run(self$tf_session$polyak$average(self$tf_session$dlt_vec)))
      } else {
        return(private$run(self$tf_session$dlt_vec))
      }
    },
    delta_idx     = function() { which(private$run(self$tf_session$dlt_free) == 1) },
    delta_free    = function() {
      if (self$polyak_result) {
        return(private$run(self$tf_session$polyak$average(self$tf_session$dlt_fre)))
      } else {
        return(private$run(self$tf_session$dlt_fre))
      }
    },
    delta_grad    = function() {
      # we need to polyak average the gradient always with SGD!
      private$run(self$tf_session$polyak$average(self$tf_session$dlt_g[[1]]))

    },
    delta_hess    = function() {
      # we need to polyak average the hessian always with SGD!
      private$run(self$tf_session$polyak$average(self$tf_session$dlt_H[[1]]))
    },
    ACOV          = function() {
      # only valid with ml_loss, uses polyak averaged hessian.
      hes <- self$delta_hess
      idx <- self$delta_idx
      (1 / (self$sample_size - 1)) * solve(hes[idx, idx])
    },
    adam_variance = function() {
      vars <- private$run(self$tf_session$optim$get_slot(self$tf_session$dlt_vec, "v"))
      vars[self$delta_idx]
    }
  )
)
