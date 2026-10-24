use pyo3::prelude::*;

/// TODO
#[pymodule(name = "imgal")]
mod imgal_module {
    #[pymodule_export]
    use crate::child_modules::colocalization_module::colocalization;
    #[pymodule_export]
    use crate::child_modules::copy_module::copy;
    #[pymodule_export]
    use crate::child_modules::distribution_module::distribution;
    #[pymodule_export]
    use crate::child_modules::filter_module::filter;
    #[pymodule_export]
    use crate::child_modules::image_module::image;
    #[pymodule_export]
    use crate::child_modules::integration_module::integration;
    #[pymodule_export]
    use crate::child_modules::kernel_module::kernel;
}
