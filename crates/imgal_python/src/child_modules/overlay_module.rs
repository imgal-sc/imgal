use pyo3::prelude::*;

/// Image overlay functions.
///
/// This module provides functions for creating and applying image overlays
/// (*e.g* a grid).
#[pymodule(submodule)]
pub mod overlay {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::child_modules::overlay_module::grid;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.overlay")
    }
}

/// Grid overlay functions.
///
/// This module provides grid overlay functions in 2D.
#[pymodule(submodule)]
pub mod grid {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::overlay_functions::grid_grid_2d_mut;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.overlay.grid")
    }
}
