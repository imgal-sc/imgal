//! Sorting, correlation and other image statistics.
//!
//! This module provides various image related statistics
//! (*e.g.* min, max, etc...), sorting and correlation functions.

mod correlation;
mod min_max;
mod percentile;
mod sample;
mod sort;
mod sum;

pub use correlation::*;
pub use min_max::*;
pub use percentile::*;
pub use sample::*;
pub use sort::*;
pub use sum::*;
