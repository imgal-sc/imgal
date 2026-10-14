//! Image filtering functions.
//!
//! This module provides *n*-dimensional image filtering functions using various
//! techniques like convolution.

mod convolve;

pub use convolve::{fft_convolve_1d, fft_deconvolve_1d};
