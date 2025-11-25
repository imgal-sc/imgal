use pyo3::prelude::*;

use super::child_modules::{
    colocalization_module, distribution_module, filter_module, image_module, integration_module,
    kernel_module, overlay_module, parameter_module, phasor_module, simulation_module,
    statistics_module, threshold_module, transform_module,
};

/// Python binding for the imgal parent module.
#[pymodule(name = "imgal")]
fn imgal_parent_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // register child modules
    colocalization_module::register_colocalization_module(m)?;
    distribution_module::register_distribution_module(m)?;
    filter_module::register_filter_module(m)?;
    image_module::register_image_module(m)?;
    integration_module::register_integration_module(m)?;
    kernel_module::register_kernel_module(m)?;
    overlay_module::register_overlay_module(m)?;
    parameter_module::register_parameter_module(m)?;
    phasor_module::register_phasor_module(m)?;
    simulation_module::register_simulation_module(m)?;
    statistics_module::register_statistics_module(m)?;
    threshold_module::register_threshold_module(m)?;
    transform_module::register_transform_module(m)?;
    Ok(())
}
