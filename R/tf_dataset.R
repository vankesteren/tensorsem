#' create tensorflow dataset
#'
#' @importFrom utils write.csv
#'
#' @keywords internal

create_tf_data <- function(dat) {

  # Information
  n_cores  <- parallel::detectCores()
  data_loc <- tempfile("tfdata_")
  mask_loc <- tempfile("tfmask_")
  n_row    <- nrow(dat)
  n_col    <- ncol(dat)
  b_size   <- 1

  # boolean mask
  na_ind   <- which(is.na(dat), arr.ind = TRUE)
  mask_mat <- matrix(1L, n_row, n_col)
  mask_mat[na_ind] <- 0L

  # edited dataset
  data_mat <- dat
  data_mat[na_ind] <- 0.0

  # write both
  write.csv(data_mat, file = data_loc, row.names = FALSE)
  write.csv(mask_mat, file = mask_loc, row.names = FALSE)

  spec_data  <- tfdatasets::csv_record_spec(data_loc)
  spec_mask  <- tfdatasets::csv_record_spec(mask_loc)

  tf_mask    <- tfdatasets::text_line_dataset(mask_loc, record_spec = spec_mask, parallel_records = n_cores)
  tf_data    <- tfdatasets::text_line_dataset(data_loc, record_spec = spec_data, parallel_records = n_cores)

  tf_mask     <- tfdatasets::dataset_prepare(tf_mask, x = !!spec_mask$names, batch_size = b_size)
  tf_data     <- tfdatasets::dataset_prepare(tf_data, x = !!spec_data$names, batch_size = b_size)

  tf_both    <- tfdatasets::zip_datasets(tf_data, tf_mask)

  tf_both     <- tfdatasets::dataset_shuffle(tf_both, n_row)
  tf_both     <- tfdatasets::dataset_prefetch(tf_both, n_row)
  dat_iter   <- tfdatasets::make_iterator_initializable(tf_both)
  next_batch <- tfdatasets::iterator_get_next(dat_iter)

  return(list(data_mat   = dat,
              tf_dat     = tf_both,
              dat_iter   = dat_iter,
              next_batch = next_batch,
              data_loc   = data_loc,
              mask_loc   = mask_loc,
              n_row      = n_row,
              n_col      = n_col,
              b_size     = b_size))
}


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

