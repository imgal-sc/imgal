//! Numerical integration functions.
//!
//! This module provides numerical integration functions that work on either 1D
//! or *n*-dimensinal data.

mod rectangle;
mod simpson;

pub use rectangle::midpoint;
pub use simpson::composite_simpson;
pub use simpson::simpson;
