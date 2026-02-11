use pyo3::prelude::*;

use crate::functions::threshold_functions;
use crate::utils::py_import_module;

/// Python binding for the "threshold" submodule.
pub fn register_threshold_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let threshold_module = PyModule::new(parent_module.py(), "threshold")?;

    // add module to Python's sys.modules
    py_import_module("threshold");

    // add threshold submodule functions
    threshold_module.add_function(wrap_pyfunction!(
        threshold_functions::threshold_manual_mask,
        &threshold_module
    )?)?;
    threshold_module.add_function(wrap_pyfunction!(
        threshold_functions::threshold_otsu_mask,
        &threshold_module
    )?)?;
    threshold_module.add_function(wrap_pyfunction!(
        threshold_functions::threshold_otsu_value,
        &threshold_module
    )?)?;

    // attach to parent module
    parent_module.add_submodule(&threshold_module)
}
