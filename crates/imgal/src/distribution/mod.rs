//! Adjustable distribution functions.
pub mod cdf;
pub use cdf::inverse_normal_cdf;
pub mod gaussian;
pub use gaussian::normalized_gaussian;
