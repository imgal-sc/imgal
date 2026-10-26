use pyo3::prelude::*;

/// Image filtering functions.
///
/// This module provides *n*-dimensional image filtering functions using various
/// techniques like convolution.
#[pymodule(submodule)]
pub mod filter {
    #[pymodule_export]
    use crate::functions::filter_functions::filter_fft_convolve_1d;
    #[pymodule_export]
    use crate::functions::filter_functions::filter_fft_deconvolve_1d;
}
