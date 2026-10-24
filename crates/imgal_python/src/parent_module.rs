use pyo3::prelude::*;

/// TODO
#[pymodule(name = "imgal")]
mod imgal_module {
    #[pymodule_export]
    use crate::child_modules::colocalization_module::colocalization;
    #[pymodule_export]
    use crate::child_modules::copy_module::copy;
}
