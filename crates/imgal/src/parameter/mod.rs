//! Microscopy and imaging related parameter functions.
mod diffraction;
pub use diffraction::abbe_diffraction_limit;
mod omega;
pub use omega::omega;
