//! Adjustable distribution functions.

mod cdf;
mod gaussian;

pub use cdf::inverse_normal_cdf;
pub use gaussian::normalized_gaussian;
