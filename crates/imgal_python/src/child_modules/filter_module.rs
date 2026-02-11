use pyo3::prelude::*;

use crate::functions::filter_functions;
use crate::utils::py_import_module;

// Python bindings for the "filters" submodule
pub fn register_filter_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let filter_module = PyModule::new(parent_module.py(), "filter")?;

    // add module to Python's sys.modules
    py_import_module("filter");

    // add filters submodule functions
    filter_module.add_function(wrap_pyfunction!(
        filter_functions::filter_fft_convolve_1d,
        &filter_module
    )?)?;
    filter_module.add_function(wrap_pyfunction!(
        filter_functions::filter_fft_deconvolve_1d,
        &filter_module
    )?)?;

    // attach to parent module
    parent_module.add_submodule(&filter_module)
}
