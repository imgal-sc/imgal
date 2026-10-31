use pyo3::prelude::*;

/// Image threshold functions.
///
/// This module provides various global and manual image threshold functions.
#[pymodule(submodule)]
pub mod threshold {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::threshold_functions::threshold_otsu_mask;
    #[pymodule_export]
    use crate::functions::threshold_functions::threshold_otsu_value;
    #[pymodule_export]
    use crate::functions::threshold_functions::threshold_manual_mask;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.threshold")
    }
}
