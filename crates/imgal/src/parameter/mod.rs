//! Microscopy and imaging related parameter functions.

mod diffraction;
mod omega;

pub use diffraction::abbe_diffraction_limit;
pub use omega::omega;
