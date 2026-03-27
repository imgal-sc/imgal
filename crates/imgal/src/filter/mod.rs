//! Filter functions.
pub(crate) mod convolve;
pub use convolve::{fft_convolve_1d, fft_deconvolve_1d};
