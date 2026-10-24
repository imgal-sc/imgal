//! Basic image processing functions.
//!
//! This module provides *n*-dimensional basic image functions such as
//! histograms and image normalization.

mod histogram;
mod normalization;

pub use histogram::histogram;
pub use histogram::histogram_bin_midpoint;
pub use histogram::histogram_bin_range;
pub use normalization::percentile_normalize;
