context("test-tf_sem")

tf_mod <- tf_sem("x1 ~ x2 + x3", lavaan::HolzingerSwineford1939[7:12])

test_that("model is created", {
  expect_equal(class(tf_mod), c("tf_sem", "R6"))
})

test_that("training works", {
  tf_mod$train(25)
  expect_equal(length(tf_mod$loss_vec), 26)
  expect_lt(tf_mod$loss, 540)
})

