use pyo3::prelude::*;

/// Probability distribution functions.
///
/// This module provides functions for creating normalized distributions and
/// computing cumulative distribution values.
#[pymodule(submodule)]
pub mod distribution {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::distribution_functions::distribution_inverse_normal_cdf;
    #[pymodule_export]
    use crate::functions::distribution_functions::distribution_normalized_gaussian;
    
    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.distribution")
    }
}
