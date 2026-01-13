use pyo3::prelude::*;

use crate::functions::copy_functions;
use crate::utils::py_import_module;

/// Python binding for the "copy" submodule.
pub fn register_copy_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let copy_module = PyModule::new(parent_module.py(), "copy")?;
    py_import_module("copy");
    copy_module.add_function(wrap_pyfunction!(
        copy_functions::copy_duplicate,
        &copy_module
    )?)?;

    parent_module.add_submodule(&copy_module)
}
