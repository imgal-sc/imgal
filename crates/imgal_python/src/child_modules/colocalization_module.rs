use pyo3::prelude::*;

use crate::functions::colocalization_functions;
use crate::utils::py_import_module;

/// Python binding for the "colocalization" submodule.
pub fn register_colocalization_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let colocalization_module = PyModule::new(parent_module.py(), "colocalization")?;

    // add module to python's sys.modules
    py_import_module("colocalization");

    // add colocalization submodule functions
    colocalization_module.add_function(wrap_pyfunction!(
        colocalization_functions::colocalization_saca_2d,
        &colocalization_module
    )?)?;
    colocalization_module.add_function(wrap_pyfunction!(
        colocalization_functions::colocalization_saca_3d,
        &colocalization_module
    )?)?;
    colocalization_module.add_function(wrap_pyfunction!(
        colocalization_functions::colocalization_saca_significance_mask,
        &colocalization_module
    )?)?;

    // attach to parent module
    parent_module.add_submodule(&colocalization_module)
}
