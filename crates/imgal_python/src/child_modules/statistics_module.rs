use pyo3::prelude::*;

use crate::functions::statistics_functions;
use crate::utils::py_import_module;

/// Python binding for the "statistics" submodule.
pub fn register_statistics_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let statistics_module = PyModule::new(parent_module.py(), "statistics")?;

    // add module to python's sys.modules
    py_import_module("statistics");

    // add statistics submodule functions
    statistics_module.add_function(wrap_pyfunction!(
        statistics_functions::statistics_effective_sample_size,
        &statistics_module
    )?)?;
    statistics_module.add_function(wrap_pyfunction!(
        statistics_functions::statistics_max,
        &statistics_module
    )?)?;
    statistics_module.add_function(wrap_pyfunction!(
        statistics_functions::statistics_min,
        &statistics_module
    )?)?;
    statistics_module.add_function(wrap_pyfunction!(
        statistics_functions::statistics_min_max,
        &statistics_module
    )?)?;
    statistics_module.add_function(wrap_pyfunction!(
        statistics_functions::statistics_linear_percentile,
        &statistics_module
    )?)?;
    statistics_module.add_function(wrap_pyfunction!(
        statistics_functions::statistics_sum,
        &statistics_module
    )?)?;
    statistics_module.add_function(wrap_pyfunction!(
        statistics_functions::statistics_weighted_kendall_tau_b,
        &statistics_module
    )?)?;
    statistics_module.add_function(wrap_pyfunction!(
        statistics_functions::statistics_weighted_merge_sort_mut,
        &statistics_module
    )?)?;

    // attach to parent module
    parent_module.add_submodule(&statistics_module)
}
