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
    penalties     = list(
      lasso_beta   = 0.0,
      lasso_lambda = 0.0,
      lasso_psi    = 0.0,
      ridge_beta   = 0.0,
      ridge_lambda = 0.0,
      ridge_psi    = 0.0
    ),
    feed          = NULL,
    initialize    = function(tf_session, mod, sample_size) {
      self$tf_session  <- tf_session
      self$lav_model   <- mod
      self$sample_size <- sample_size
      private$update_feed()
      self$loss_vec    <- self$loss
    },

    # Methods
    train         = function(niter = 10000, pb = TRUE, verbose = FALSE) {
      private$update_feed()

      loss_vec <- numeric(niter)

      if (verbose) {
        cat(str_pad("ITER", nchar(niter)), " | ", str_pad("LOSS", nchar(format(round(self$loss, 5)))), " | ",
            "FIRST 10 FREE PARAMETERS\n")
      } else if (pb) {
        progbar <- progress_bar$new(format = "[loss: :loss] [:bar] :percent", total  = niter)
      }

      for (iter in 1:niter) {
        private$run(self$tf_session$train)

        loss_vec[iter] <- self$loss

        if (pb && !verbose) progbar$tick(tokens = list(loss = format(round(loss_vec[iter], 5), nsmall = 5)))

        if (verbose && iter %% ceiling(niter / 100) == 0) {
          freeparams <- self$delta_free
          if (length(freeparams) > 10) freeparams <- freeparams[1:10]

          cat(str_pad(iter, nchar(niter)), " | ",
              format(round(self$loss, 5), nsmall = 5), " | ",
              format(round(freeparams,  3), nsmall = 3), "\n")
        }
      }

      self$loss_vec <- c(self$loss_vec, loss_vec)

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
      loss <- tryCatch(private$run(self$tf_session$loss), error = function(e) Inf)
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
    plot_loss     = function() {
      if (length(self$loss_vec) < 2) stop("Too few iterations to plot loss.")
      plot(x    = 1:length(self$loss_vec),
           y    = self$loss_vec,
           xlab = "Iterations",
           ylab = "Loss",
           main = "Loss plot",
           bty  = "L",
           type = "l",
           col  = "#00008b"
      )
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
    Sigma         = function() { private$run(self$tf_session$Sigma) },
    Psi           = function() { private$run(self$tf_session$Psi) },
    Beta          = function() { private$run(self$tf_session$B_0) },
    Lambda        = function() { private$run(self$tf_session$Lambda) },
    Theta         = function() { private$run(self$tf_session$Theta) },

    # gradients
    Psi_grad      = function() { private$run(self$tf_session$Psi_g) },
    Beta_grad     = function() { private$run(self$tf_session$B_0_g) },
    Lambda_grad   = function() { private$run(self$tf_session$Lambda_g) },
    Theta_grad    = function() { private$run(self$tf_session$Theta_g) },

    # data & loss
    data          = function() {
      dat <- private$run(self$tf_session$S)
      colnames(dat) <- rownames(dat) <- self$tf_session$v_names[self$tf_session$v_trans]
      dat
    },
    loss          = function() { private$run(self$tf_session$loss) },
    loglik        = function() { (-(self$sample_size - 1) / 2) * (ncol(self$data) * log(2 * pi) + self$loss) },

    # param vec
    delta         = function() { private$run(self$tf_session$dlt_vec) },
    delta_idx     = function() { which(private$run(self$tf_session$dlt_free) == 1) },
    delta_free    = function() { private$run(self$tf_session$dlt_fre) },
    delta_grad    = function() { private$run(self$tf_session$dlt_g) },
    delta_hess    = function() { private$run(self$tf_session$dlt_H)[[1]] },
    ACOV          = function() {
      idx <- self$delta_idx
      hes <- self$delta_hess
      (2 / (self$sample_size - 1)) * solve(hes[idx, idx])
    }
  )
)
