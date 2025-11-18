use pyo3::prelude::*;

use crate::functions::image_functions;
use crate::utils::py_import_module;

/// Python bindings for the "image" submodule.
pub fn register_image_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let image_module = PyModule::new(parent_module.py(), "image")?;

    // add module to Python's sys.modules
    py_import_module("image");

    // add image submodule functions
    image_module.add_function(wrap_pyfunction!(
        image_functions::image_histogram,
        &image_module
    )?)?;
    image_module.add_function(wrap_pyfunction!(
        image_functions::image_histogram_bin_midpoint,
        &image_module
    )?)?;
    image_module.add_function(wrap_pyfunction!(
        image_functions::image_histogram_bin_range,
        &image_module
    )?)?;
    image_module.add_function(wrap_pyfunction!(
        image_functions::normalize_percentile_normalize,
        &image_module
    )?)?;

    // attach to parent module
    parent_module.add_submodule(&image_module)
}
