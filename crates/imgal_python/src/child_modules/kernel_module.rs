use pyo3::prelude::*;

/// Kernel and neighborhood functions.
///
/// This module provides 2D and 3D kernels, weighted kernels and other
/// neighborhood related functions.
#[pymodule(submodule)]
pub mod kernel {
    #[pymodule_export]
    use crate::child_modules::kernel_module::neighborhood;
}

/// Neighborhood functions.
///
/// This module provides weighted and non-weighted neighborhood kernels in 2D
/// and 3D.
#[pymodule(submodule)]
pub mod neighborhood {
    #[pymodule_export]
    use crate::functions::kernel_functions::neighborhood_circle_kernel;
    #[pymodule_export]
    use crate::functions::kernel_functions::neighborhood_sphere_kernel;
    #[pymodule_export]
    use crate::functions::kernel_functions::neighborhood_weighted_circle_kernel;
    #[pymodule_export]
    use crate::functions::kernel_functions::neighborhood_weighted_sphere_kernel;
}
