use pyo3::prelude::*;

use crate::functions::distribution_functions;
use crate::utils::py_import_module;

/// Python bindings for the "distribution" submodule
pub fn register_distribution_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let distribution_module = PyModule::new(parent_module.py(), "distribution")?;

    // add module to Python's sys.modules
    py_import_module("distribution");

    // add distribution submodule functions
    distribution_module.add_function(wrap_pyfunction!(
        distribution_functions::distribution_inverse_normal_cdf,
        &distribution_module
    )?)?;
    distribution_module.add_function(wrap_pyfunction!(
        distribution_functions::distribution_normalized_gaussian,
        &distribution_module
    )?)?;

    // attach to parent module
    parent_module.add_submodule(&distribution_module)
}
