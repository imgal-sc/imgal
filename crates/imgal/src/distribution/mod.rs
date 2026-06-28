//! Probability distribution functions.
//!
//! This module provides functions for creating normalized distributions and
//! computing cumulative distribution values.

mod cdf;
mod gaussian;

pub use cdf::inverse_normal_cdf;
pub use gaussian::normalized_gaussian;
