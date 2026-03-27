//! Adjustable distribution functions.
pub(crate) mod cdf;
pub use cdf::inverse_normal_cdf;
pub(crate) mod gaussian;
pub use gaussian::normalized_gaussian;
