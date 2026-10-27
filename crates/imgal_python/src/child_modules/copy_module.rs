use pyo3::prelude::*;

/// Copy *n*-dimensional image data.
///
/// This module provides *n*-dimensional image copying functions where data can
/// be duplicated, copied into pre-existing containers (*i.e.* arrays) and
/// copied into 1D flat arrays.
#[pymodule(submodule)]
pub mod copy {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::copy_functions::copy_copy_into;
    #[pymodule_export]
    use crate::functions::copy_functions::copy_copy_into_flat;
    #[pymodule_export]
    use crate::functions::copy_functions::copy_duplicate;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.copy")
    }
}
