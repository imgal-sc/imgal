use pyo3::prelude::*;

use crate::functions::kernel_functions;
use crate::utils::py_import_module;

/// Python bindings for the "kernel" submodule.
pub fn register_kernel_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let kernel_module = PyModule::new(parent_module.py(), "kernel")?;
    let neighborhood_module = PyModule::new(parent_module.py(), "neighborhood")?;

    // add module to Python's sys.modules
    py_import_module("kernel");
    py_import_module("kernel.neighborhood");

    // add kernel::neighborhood submodule functions
    neighborhood_module.add_function(wrap_pyfunction!(
        kernel_functions::neighborhood_circle_kernel,
        &neighborhood_module
    )?)?;
    neighborhood_module.add_function(wrap_pyfunction!(
        kernel_functions::neighborhood_sphere_kernel,
        &neighborhood_module
    )?)?;
    neighborhood_module.add_function(wrap_pyfunction!(
        kernel_functions::neighborhood_weighted_circle_kernel,
        &neighborhood_module
    )?)?;
    neighborhood_module.add_function(wrap_pyfunction!(
        kernel_functions::neighborhood_weighted_sphere_kernel,
        &neighborhood_module
    )?)?;

    // attach kernel submodules before attaching to the parent module
    kernel_module.add_submodule(&neighborhood_module)?;
    parent_module.add_submodule(&kernel_module)
}
