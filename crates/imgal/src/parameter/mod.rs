//! Microscopy and imaging related parameter functions.
pub mod diffraction;
pub use diffraction::abbe_diffraction_limit;

pub mod omega;
pub use omega::omega;
