use pyo3::prelude::*;

/// Numerical integration functions.
///
/// This module provides numerical integration functions that work on either 1D
/// or *n*-dimensinal data.
#[pymodule(submodule)]
pub mod integration {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::integration_functions::integration_composite_simpson;
    #[pymodule_export]
    use crate::functions::integration_functions::integration_midpoint;
    #[pymodule_export]
    use crate::functions::integration_functions::integration_simpson;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.integration")
    }
}
