use pyo3::prelude::*;

use crate::functions::spatial_functions;
use crate::utils::py_import_module;

/// Python binding for the "spatial" submodule.
pub fn register_spatial_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let spatial_module = PyModule::new(parent_module.py(), "spatial")?;
    let convex_hull_module = PyModule::new(parent_module.py(), "convex_hull")?;
    let roi_module = PyModule::new(parent_module.py(), "roi")?;
    py_import_module("spatial");
    py_import_module("spatial.convex_hull");
    py_import_module("spatial.roi");
    convex_hull_module.add_function(wrap_pyfunction!(
        spatial_functions::spatial_chan_2d,
        &convex_hull_module
    )?)?;
    convex_hull_module.add_function(wrap_pyfunction!(
        spatial_functions::spatial_graham_scan,
        &convex_hull_module
    )?)?;
    convex_hull_module.add_function(wrap_pyfunction!(
        spatial_functions::spatial_jarvis_march,
        &convex_hull_module
    )?)?;
    convex_hull_module.add_function(wrap_pyfunction!(
        spatial_functions::spatial_preparata_hong_3d,
        &convex_hull_module
    )?)?;
    roi_module.add_function(wrap_pyfunction!(
        spatial_functions::spatial_roi_cloud_map,
        &roi_module
    )?)?;
    roi_module.add_function(wrap_pyfunction!(
        spatial_functions::spatial_roi_data_map,
        &roi_module
    )?)?;
    spatial_module.add_submodule(&convex_hull_module)?;
    spatial_module.add_submodule(&roi_module)?;
    parent_module.add_submodule(&spatial_module)
}
