% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/helpers.R
\name{vech_dup_idx}
\alias{vech_dup_idx}
\title{Constructs index vector for transforming a vech vector
into a vec vector to create an n*n symmetric matrix
from the vech vector.
tensor$index_select(0, idx)$view(3,3)}
\usage{
vech_dup_idx(n)
}
\arguments{
\item{n}{size of the resulting square matrix}
}
\value{
array containing the indices
}
\description{
Constructs index vector for transforming a vech vector
into a vec vector to create an n*n symmetric matrix
from the vech vector.
tensor$index_select(0, idx)$view(3,3)
}
