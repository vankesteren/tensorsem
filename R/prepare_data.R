#' Prepare data for tensorsem model
#'
#' This function prepares a dataframe for a tensorsem model. It
#' first converts the variables to a design matrix, then centers
#' it, and lastly converts it to a torch_tensor
#'
#' @param df data frame
#' @param dtype data type of the resulting tensor
#' @param device device to store the resulting tensor on
#'
#' @return Torch tensor of scaled and processed data
#'
#' @seealso [torch::torch_tensor()], [stats::model.matrix()]
#'
#' @export
df_to_tensor <- function(df, dtype = NULL, device = NULL) {
  torch::torch_tensor(
    data = scale(model.matrix(~ . - 1, df), scale = FALSE),
    requires_grad = FALSE,
    dtype = dtype,
    device = device
  )
}
