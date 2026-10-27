use pyo3::prelude::*;

/// Microscopy and imaging related parameter functions.
///
/// This module provides functions for calculating common microscopy imaging
/// parameters.
#[pymodule(submodule)]
pub mod parameter {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::parameter_functions::parameter_abbe_diffraction_limit;
    #[pymodule_export]
    use crate::functions::parameter_functions::parameter_omega;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.parameter")
    }
}
