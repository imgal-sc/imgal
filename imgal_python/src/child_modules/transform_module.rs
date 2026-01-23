use pyo3::prelude::*;

use crate::functions::transform_functions;
use crate::utils::py_import_module;

/// Python binding for the "transform" submodule.
pub fn register_transform_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let transform_module = PyModule::new(parent_module.py(), "transform")?;
    let pad_module = PyModule::new(parent_module.py(), "pad")?;
    let tile_module = PyModule::new(parent_module.py(), "tile")?;
    py_import_module("transform");
    py_import_module("transform.pad");
    py_import_module("transform.tile");
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
    tile_module.add_function(wrap_pyfunction!(
        transform_functions::tile_div_tile,
        &tile_module
    )?)?;
    tile_module.add_function(wrap_pyfunction!(
        transform_functions::tile_div_untile,
        &tile_module
    )?)?;

    transform_module.add_submodule(&pad_module)?;
    transform_module.add_submodule(&tile_module)?;

    parent_module.add_submodule(&transform_module)
}
