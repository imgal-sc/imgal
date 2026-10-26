use pyo3::prelude::*;

/// Image transformation functions.
///
/// This module provides various image transformation functions (*e.g*
/// reflect padding, sum projection, etc...).
#[pymodule(submodule)]
pub mod transform {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::child_modules::transform_module::pad;
    #[pymodule_export]
    use crate::child_modules::transform_module::project;
    #[pymodule_export]
    use crate::child_modules::transform_module::tile;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.transform")
    }
}

/// Image padding functions.
///
/// This module provides *n*-dimensional image padding functions.
#[pymodule(submodule)]
pub mod pad {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::transform_functions::pad_constant_pad;
    #[pymodule_export]
    use crate::functions::transform_functions::pad_reflect_pad;
    #[pymodule_export]
    use crate::functions::transform_functions::pad_zero_pad;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.transform.pad")
    }
}

/// Image projection functions.
///
/// This module provides *n*-dimensional image projection functions.
#[pymodule(submodule)]
pub mod project {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::transform_functions::project_sum_project;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.transform.project")
    }
}
/// Image tiling functions.
///
/// This module provides *n*-dimensional image tiling functions.
#[pymodule(submodule)]
pub mod tile {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::transform_functions::tile_div_tile;
    #[pymodule_export]
    use crate::functions::transform_functions::tile_div_untile;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.transform.tile")
    }
}
