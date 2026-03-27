//! Microscopy and imaging related parameter functions.
pub(crate) mod diffraction;
pub use diffraction::abbe_diffraction_limit;
pub(crate) mod omega;
pub use omega::omega;
