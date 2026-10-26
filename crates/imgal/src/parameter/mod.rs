//! Microscopy and imaging related parameter functions.
//!
//! This module provides functions for calculating common microscopy imaging
//! parameters.

mod diffraction;
mod omega;

pub use diffraction::abbe_diffraction_limit;
pub use omega::omega;
