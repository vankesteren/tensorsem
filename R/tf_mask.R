# how does the mask operator work?


# library(tensorflow)
#
# mask <- tf$placeholder("int32",   shape = shape(3), name = "mask")
# data <- tf$placeholder("float32", shape = shape(NULL, 3), name = "data")
# data_masked <- tf$boolean_mask(data, mask, axis = 1L)
#
#
# S    <- tf$matmul(data_masked, data_masked, transpose_a = TRUE, name = "outprod")
#
# Sigma <- tf$Variable(matrix(1:9, 3, 3), dtype = "float32", name = "Sigma")
# Sigma_masked <- tf$boolean_mask(tf$boolean_mask(Sigma, mask, axis = 0L), mask, axis = 1L)
#
# S_mask <- tf$reshape(tf$boolean_mask(Sigma, tf$matmul(mask, mask, transpose_a = TRUE)), shape = tf$shape(S))
#
# sess <- tf$Session()
# sess$run(tf$global_variables_initializer())
#
# msk <- matrix(c(rep(0L, 4), rep(1L, 8)), 4)
# dat <- matrix(rnorm(12), 4)
#
# sess$run(c(mask, data, data_masked, S, Sigma, Sigma_masked),
#          feed_dict = dict("mask:0" = c(0L, 1L, 1L),
#                           "data:0" = dat))
#
#
