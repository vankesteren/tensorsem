% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/helpers.R
\name{torch_jacobian}
\alias{torch_jacobian}
\title{Compute jacobian of output wrt input tensor}
\usage{
torch_jacobian(output, input)
}
\arguments{
\item{output}{Tensor vector of size Po}

\item{input}{Tensor vector of size Pi}
}
\value{
jacobian: Tensor of size Pi, Po
}
\description{
Compute jacobian of output wrt input tensor
}
