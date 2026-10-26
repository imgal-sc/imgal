use pyo3::prelude::*;

/// Probability distribution functions.
///
/// This module provides functions for creating normalized distributions and
/// computing cumulative distribution values.
#[pymodule(submodule)]
pub mod distribution {
    #[pymodule_export]
    use crate::functions::distribution_functions::distribution_inverse_normal_cdf;
    #[pymodule_export]
    use crate::functions::distribution_functions::distribution_normalized_gaussian;
}
