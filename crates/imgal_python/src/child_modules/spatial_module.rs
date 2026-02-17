use pyo3::prelude::*;

use crate::functions::spatial_functions;
use crate::utils::py_import_module;

/// Python binding for the "spatial" submodule.
pub fn register_spatial_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let spatial_module = PyModule::new(parent_module.py(), "spatial")?;
    py_import_module("spatial");
    spatial_module.add_function(wrap_pyfunction!(
        spatial_functions::spatial_roi_map,
        &spatial_module
    )?)?;
    parent_module.add_submodule(&spatial_module)
}
