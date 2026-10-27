use pyo3::prelude::*;

/// Image filtering functions.
///
/// This module provides *n*-dimensional image filtering functions using various
/// techniques like convolution.
#[pymodule(submodule)]
pub mod filter {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::filter_functions::filter_fft_convolve_1d;
    #[pymodule_export]
    use crate::functions::filter_functions::filter_fft_deconvolve_1d;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.filter")
    }    
}
