use pyo3::prelude::*;

use crate::functions::phasor_functions;
use crate::utils::py_import_module;

/// Python binding for the "phasor" submodule.
pub fn register_phasor_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let phasor_module = PyModule::new(parent_module.py(), "phasor")?;
    let calibration_module = PyModule::new(parent_module.py(), "calibration")?;
    let plot_module = PyModule::new(parent_module.py(), "plot")?;
    let time_domain_module = PyModule::new(parent_module.py(), "time_domain")?;
    py_import_module("phasor");
    py_import_module("phasor.calibration");
    py_import_module("phasor.plot");
    py_import_module("phasor.time_domain");
    calibration_module.add_function(wrap_pyfunction!(
        phasor_functions::calibration_calibrate_coords,
        &calibration_module
    )?)?;
    calibration_module.add_function(wrap_pyfunction!(
        phasor_functions::calibration_calibrate_gs_image,
        &calibration_module
    )?)?;
    calibration_module.add_function(wrap_pyfunction!(
        phasor_functions::calibration_calibrate_gs_image_mut,
        &calibration_module
    )?)?;
    calibration_module.add_function(wrap_pyfunction!(
        phasor_functions::calibration_modulation_and_phase,
        &calibration_module
    )?)?;
    plot_module.add_function(wrap_pyfunction!(
        phasor_functions::plot_gs_mask,
        &plot_module
    )?)?;
    plot_module.add_function(wrap_pyfunction!(
        phasor_functions::plot_gs_modulation,
        &plot_module
    )?)?;
    plot_module.add_function(wrap_pyfunction!(
        phasor_functions::plot_gs_phase,
        &plot_module
    )?)?;
    plot_module.add_function(wrap_pyfunction!(
        phasor_functions::plot_monoexponential_coords,
        &plot_module
    )?)?;
    time_domain_module.add_function(wrap_pyfunction!(
        phasor_functions::time_domain_gs_image,
        &time_domain_module
    )?)?;
    time_domain_module.add_function(wrap_pyfunction!(
        phasor_functions::time_domain_gs_roi,
        &time_domain_module
    )?)?;
    time_domain_module.add_function(wrap_pyfunction!(
        phasor_functions::time_domain_imaginary_coord,
        &time_domain_module
    )?)?;
    time_domain_module.add_function(wrap_pyfunction!(
        phasor_functions::time_domain_real_coord,
        &time_domain_module
    )?)?;
    phasor_module.add_submodule(&calibration_module)?;
    phasor_module.add_submodule(&plot_module)?;
    phasor_module.add_submodule(&time_domain_module)?;
    parent_module.add_submodule(&phasor_module)
}
