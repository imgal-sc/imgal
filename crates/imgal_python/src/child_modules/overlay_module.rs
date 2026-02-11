use pyo3::prelude::*;

use crate::functions::overlay_functions;
use crate::utils::py_import_module;

/// Python binding for the "overlay" submodule.
pub fn register_overlay_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let overlay_module = PyModule::new(parent_module.py(), "overlay")?;
    let grid_module = PyModule::new(parent_module.py(), "grid")?;

    // add module to Python's sys.modules
    py_import_module("overlay");
    py_import_module("overlay.grid");

    // add overlay submodule functions
    grid_module.add_function(wrap_pyfunction!(
        overlay_functions::grid_grid_2d_mut,
        &grid_module
    )?)?;

    // attach overlay submodules before attaching to the parent module
    overlay_module.add_submodule(&grid_module)?;
    parent_module.add_submodule(&overlay_module)
}
