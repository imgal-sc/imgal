use pyo3::prelude::*;

/// Image threshold functions.
///
/// This module provides various global and manual image threshold functions.
#[pymodule(submodule)]
pub mod threshold {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::child_modules::threshold_module::global;
    #[pymodule_export]
    use crate::child_modules::threshold_module::manual;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.threshold")
    }
}

/// Global image threshold functions.
///
/// This module provides global image threshold functions for calculating
/// threshold values and creating masks.
#[pymodule(submodule)]
pub mod global {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::threshold_functions::global_otsu_mask;
    #[pymodule_export]
    use crate::functions::threshold_functions::global_otsu_value;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.threshold.global")
    }
}

/// Manual image threshold functions.
///
/// This module provides manual image threshold functions for creating masks.
#[pymodule(submodule)]
pub mod manual {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::threshold_functions::manual_manual_mask;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.threshold.manual")
    }
}
