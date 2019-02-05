create_tf_data <- function(dat) {

  # Information
  n_cores <- parallel::detectCores()
  tmp_loc <- tempfile("tfdata_")
  n_row   <- nrow(dat)
  n_col   <- ncol(dat)
  b_size  <- 1


  write.csv(dat, file = tmp_loc, row.names = FALSE)
  dat_spec   <- tfdatasets::csv_record_spec(tmp_loc)
  tf_dat     <- tfdatasets::text_line_dataset(tmp_loc, record_spec = dat_spec, parallel_records = n_cores)
  tf_dat     <- tfdatasets::dataset_shuffle(tf_dat, n_row)
  tf_dat     <- tfdatasets::dataset_prepare(tf_dat, x = !!colnames(dat), batch_size = b_size)
  tf_dat     <- tfdatasets::dataset_prefetch(tf_dat, n_row)
  dat_iter   <- tfdatasets::make_iterator_initializable(tf_dat)
  next_batch <- tfdatasets::iterator_get_next(dat_iter)

  return(list(tf_dat     = tf_dat,
              dat_iter   = dat_iter,
              next_batch = next_batch,
              loc        = tmp_loc,
              n_row      = n_row,
              n_col      = n_col,
              b_size     = b_size))
}
#
# sess <- tf$Session()
# sess$run(tf$global_variables_initializer())
# sess$run(ok$dat_iter$initializer)
#
# tfdatasets::until_out_of_range({
#   batch <- sess$run(next_batch)
#   str(batch)
# })
#
