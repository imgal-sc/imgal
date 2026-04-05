//! Image functions.
pub(crate) mod histogram;
pub use histogram::histogram;
pub use histogram::histogram_bin_midpoint;
pub use histogram::histogram_bin_range;
pub(crate) mod normalization;
pub use normalization::percentile_normalize;
