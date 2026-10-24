use pyo3::prelude::*;

/// Numerical integration functions.
///
/// This module provides numerical integration functions that work on either 1D
/// or *n*-dimensinal data.
#[pymodule(submodule)]
pub mod integration {
    #[pymodule_export]
    use crate::functions::integration_functions::integration_composite_simpson;
    #[pymodule_export]
    use crate::functions::integration_functions::integration_midpoint;
    #[pymodule_export]
    use crate::functions::integration_functions::integration_simpson;
}

