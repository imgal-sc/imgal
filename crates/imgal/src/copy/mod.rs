//! Copy n-dimensional image data.
//!
//! This module provides *n*-dimensional image copying functions where data can
//! be duplicated, copied into pre-existing containers (*i.e.* arrays) and
//! copied into 1D flat arrays.

mod duplicate;

pub use duplicate::copy_into;
pub use duplicate::copy_into_flat;
pub use duplicate::duplicate;
