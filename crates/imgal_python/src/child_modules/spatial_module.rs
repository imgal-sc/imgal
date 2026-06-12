use pyo3::prelude::*;

use crate::functions::spatial_functions;
use crate::utils::py_import_module;

/// Python binding for the "spatial" submodule.
pub fn register_spatial_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let spatial_module = PyModule::new(parent_module.py(), "spatial")?;
    let convex_hull_module = PyModule::new(parent_module.py(), "convex_hull")?;
    let geometry_module = PyModule::new(parent_module.py(), "geometry")?;
    let halfspace_module = PyModule::new(parent_module.py(), "halfspace")?;
    let roi_module = PyModule::new(parent_module.py(), "roi")?;
    py_import_module("spatial");
    py_import_module("spatial.convex_hull");
    py_import_module("spatial.geometry");
    py_import_module("spatial.halfspace");
    py_import_module("spatial.roi");
    convex_hull_module.add_function(wrap_pyfunction!(
        spatial_functions::convex_hull_chan_2d,
        &convex_hull_module
    )?)?;
    convex_hull_module.add_function(wrap_pyfunction!(
        spatial_functions::convex_hull_graham_scan,
        &convex_hull_module
    )?)?;
    convex_hull_module.add_function(wrap_pyfunction!(
        spatial_functions::convex_hull_jarvis_march,
        &convex_hull_module
    )?)?;
    convex_hull_module.add_function(wrap_pyfunction!(
        spatial_functions::convex_hull_quickhull_3d,
        &convex_hull_module
    )?)?;
    geometry_module.add_function(wrap_pyfunction!(
        spatial_functions::geometry_inside_polyhedron,
        &geometry_module
    )?)?;
    geometry_module.add_function(wrap_pyfunction!(
        spatial_functions::geometry_inside_tetrahedron,
        &geometry_module
    )?)?;
    geometry_module.add_function(wrap_pyfunction!(
        spatial_functions::geometry_orient_pred_2d,
        &geometry_module
    )?)?;
    geometry_module.add_function(wrap_pyfunction!(
        spatial_functions::geometry_orient_pred_3d,
        &geometry_module
    )?)?;
    geometry_module.add_function(wrap_pyfunction!(
        spatial_functions::geometry_polyhedron_volume,
        &geometry_module
    )?)?;
    geometry_module.add_function(wrap_pyfunction!(
        spatial_functions::geometry_tetrahedron_volume,
        &geometry_module
    )?)?;
    halfspace_module.add_function(wrap_pyfunction!(
        spatial_functions::halfspace_face_to_halfspace,
        &halfspace_module
    )?)?;
    halfspace_module.add_function(wrap_pyfunction!(
        spatial_functions::halfspace_halfspace_intersection,
        &halfspace_module
    )?)?;
    halfspace_module.add_function(wrap_pyfunction!(
        spatial_functions::halfspace_hull_to_halfspace,
        &halfspace_module
    )?)?;
    halfspace_module.add_function(wrap_pyfunction!(
        spatial_functions::halfspace_inside_halfspace_inerior,
        &halfspace_module
    )?)?;
    roi_module.add_function(wrap_pyfunction!(
        spatial_functions::roi_roi_cloud_map,
        &roi_module
    )?)?;
    roi_module.add_function(wrap_pyfunction!(
        spatial_functions::roi_roi_data_map,
        &roi_module
    )?)?;
    spatial_module.add_submodule(&convex_hull_module)?;
    spatial_module.add_submodule(&geometry_module)?;
    spatial_module.add_submodule(&halfspace_module)?;
    spatial_module.add_submodule(&roi_module)?;
    parent_module.add_submodule(&spatial_module)
}
