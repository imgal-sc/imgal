use pyo3::prelude::*;

use crate::functions::parameter_functions;
use crate::utils::py_import_module;

/// Python binding for the "parameters" submodule.
pub fn register_parameter_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let parameter_module = PyModule::new(parent_module.py(), "parameter")?;

    // add module to python's sys.modules
    py_import_module("parameter");

    // add parameters submodule functions
    parameter_module.add_function(wrap_pyfunction!(
        parameter_functions::parameter_abbe_diffraction_limit,
        &parameter_module
    )?)?;
    parameter_module.add_function(wrap_pyfunction!(
        parameter_functions::parameter_omega,
        &parameter_module
    )?)?;

    // attach to parent module
    parent_module.add_submodule(&parameter_module)
}
