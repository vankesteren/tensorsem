#' create tensorflow dataset
#'
#' @importFrom utils write.csv
#' @importFrom R6 R6Class
#'
#' @keywords internal

tf_data <- R6Class(
  classname = "tf_sem",
  public = list(
    tf_dat     = NULL,
    iter       = NULL,
    get_next   = NULL,
    data_loc   = NULL,
    mask_loc   = NULL,
    n_row      = NULL,
    n_col      = NULL,
    b_size     = 1,
    initialize = function(dat) {
      # Information
      n_cores  <- parallel::detectCores()
      self$data_loc <- tempfile("tfdata_")
      self$mask_loc <- tempfile("tfmask_")
      self$n_row    <- nrow(dat)
      self$n_col    <- ncol(dat)

      # boolean mask for missing values
      na_ind   <- which(is.na(dat), arr.ind = TRUE)
      mask_mat <- matrix(1L, self$n_row, self$n_col)
      mask_mat[na_ind] <- 0L

      # edited dataset with only numeric
      data_mat <- dat
      data_mat[na_ind] <- 0.0

      # write both to temp location
      write.csv(data_mat, file = self$data_loc, row.names = FALSE)
      write.csv(mask_mat, file = self$mask_loc, row.names = FALSE)

      # prepare data as tensorflow dataset for stochastic gradient descent
      # get specifications of the csvs we just saved
      spec_data <- tfdatasets::csv_record_spec(self$data_loc)
      spec_mask <- tfdatasets::csv_record_spec(self$mask_loc)

      # generate two tf datasets with this specification
      tf_data <- tfdatasets::text_line_dataset(self$data_loc, record_spec = spec_data, parallel_records = n_cores)
      tf_mask <- tfdatasets::text_line_dataset(self$mask_loc, record_spec = spec_mask, parallel_records = n_cores)
      tf_data <- tfdatasets::dataset_prepare(tf_data, x = !!spec_data$names, batch_size = self$b_size)
      tf_mask <- tfdatasets::dataset_prepare(tf_mask, x = !!spec_mask$names, batch_size = self$b_size)

      # combine the datasets so they can be iterated simultaneously
      self$tf_dat   <- tfdatasets::zip_datasets(tf_data, tf_mask)
      self$tf_dat   <- tfdatasets::dataset_shuffle(self$tf_dat, self$n_row)
      self$tf_dat   <- tfdatasets::dataset_prefetch(self$tf_dat, min(self$n_row, 10000))
      self$iter     <- tfdatasets::make_iterator_initializable(self$tf_dat)
      self$get_next <- tfdatasets::iterator_get_next(self$iter)
    }
  )
)

# Z         <- next_batch
# Z_data    <- Z[[1]]$x
# Z_mask    <- Z[[2]]$x
# Z_only    <- tf$boolean_mask(Z_data, Z_mask)
#
# sess <- tf$Session()
# sess$run(tf$global_variables_initializer())
# sess$run(dat_iter$initializer)
#
# tfdatasets::until_out_of_range({
#   batch <- sess$run(list(Z_data, Z_mask, Z_only))
#   str(batch)
# })

