use pyo3::prelude::*;

use crate::functions::transform_functions;
use crate::utils::py_import_module;

/// Python binding for the "transform" submodule.
pub fn register_transform_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let transform_module = PyModule::new(parent_module.py(), "transform")?;
    let pad_module = PyModule::new(parent_module.py(), "pad")?;

    // add module to Python's sys.modules
    py_import_module("transform");
    py_import_module("transform.pad");

    // add threshold submodule phasor_functions
    pad_module.add_function(wrap_pyfunction!(
        transform_functions::pad_constant_pad,
        &pad_module
    )?)?;
    pad_module.add_function(wrap_pyfunction!(
        transform_functions::pad_reflect_pad,
        &pad_module
    )?)?;
    pad_module.add_function(wrap_pyfunction!(
        transform_functions::pad_zero_pad,
        &pad_module
    )?)?;

    // attach to parent module
    transform_module.add_submodule(&pad_module)?;
    parent_module.add_submodule(&transform_module)
}
