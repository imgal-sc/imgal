use pyo3::prelude::*;

/// Kernel and neighborhood functions.
///
/// This module provides 2D and 3D kernels, weighted kernels and other
/// neighborhood related functions.
#[pymodule(submodule)]
pub mod kernel {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::child_modules::kernel_module::neighborhood;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.kernel")
    }
}

/// Neighborhood functions.
///
/// This module provides weighted and non-weighted neighborhood kernels in 2D
/// and 3D.
#[pymodule(submodule)]
pub mod neighborhood {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::kernel_functions::neighborhood_circle_kernel;
    #[pymodule_export]
    use crate::functions::kernel_functions::neighborhood_sphere_kernel;
    #[pymodule_export]
    use crate::functions::kernel_functions::neighborhood_weighted_circle_kernel;
    #[pymodule_export]
    use crate::functions::kernel_functions::neighborhood_weighted_sphere_kernel;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.kernel.neighborhood")
    }
}
