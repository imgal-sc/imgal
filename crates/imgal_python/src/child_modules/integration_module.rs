use pyo3::prelude::*;

use crate::functions::integration_functions;
use crate::utils::py_import_module;

/// Python binding for the "integrate" submodule.
pub fn register_integration_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let integration_module = PyModule::new(parent_module.py(), "integration")?;

    // add module to python's sys.modules
    py_import_module("integration");

    // add integrate submodule functions
    integration_module.add_function(wrap_pyfunction!(
        integration_functions::integration_composite_simpson,
        &integration_module
    )?)?;
    integration_module.add_function(wrap_pyfunction!(
        integration_functions::integration_midpoint,
        &integration_module
    )?)?;
    integration_module.add_function(wrap_pyfunction!(
        integration_functions::integration_simpson,
        &integration_module
    )?)?;

    // attach to parent module
    parent_module.add_submodule(&integration_module)
}
