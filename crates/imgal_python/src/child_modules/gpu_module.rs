use pyo3::prelude::*;

/// GPU device setup and tensors.
///
/// This module provides the functions and structs needed to initialize a new
/// GPU device, create/read/write tensors, and convert them into GPU kernel
/// arguments.
#[pymodule(submodule)]
pub mod gpu {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::classes::gpu_classes::PyGpuTensor;
    #[pymodule_export]
    use crate::functions::gpu_functions::gpu_init_gpu;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.gpu")
    }
}
