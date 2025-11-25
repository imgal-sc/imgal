use pyo3::prelude::*;

use crate::functions::simulation_functions;
use crate::utils::py_import_module;

/// Python bindings for the "simulation" submodule.
pub fn register_simulation_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let simulation_module = PyModule::new(parent_module.py(), "simulation")?;
    let decay_module = PyModule::new(parent_module.py(), "decay")?;
    let gradient_module = PyModule::new(parent_module.py(), "gradient")?;
    let instrument_module = PyModule::new(parent_module.py(), "instrument")?;
    let noise_module = PyModule::new(parent_module.py(), "noise")?;

    // add module to python's sys.modules
    py_import_module("simulation");
    py_import_module("simulation.decay");
    py_import_module("simulation.gradient");
    py_import_module("simulation.instrument");
    py_import_module("simulation.noise");

    // add simulation::decay submodule functions
    decay_module.add_function(wrap_pyfunction!(
        simulation_functions::decay_gaussian_exponential_decay_1d,
        &decay_module
    )?)?;
    decay_module.add_function(wrap_pyfunction!(
        simulation_functions::decay_gaussian_exponential_decay_3d,
        &decay_module
    )?)?;
    decay_module.add_function(wrap_pyfunction!(
        simulation_functions::decay_ideal_exponential_decay_1d,
        &decay_module
    )?)?;
    decay_module.add_function(wrap_pyfunction!(
        simulation_functions::decay_ideal_exponential_decay_1d,
        &decay_module
    )?)?;
    decay_module.add_function(wrap_pyfunction!(
        simulation_functions::decay_ideal_exponential_decay_3d,
        &decay_module
    )?)?;
    decay_module.add_function(wrap_pyfunction!(
        simulation_functions::decay_irf_exponential_decay_1d,
        &decay_module
    )?)?;
    decay_module.add_function(wrap_pyfunction!(
        simulation_functions::decay_irf_exponential_decay_3d,
        &decay_module
    )?)?;

    // add simulation::gradient submodule functions
    gradient_module.add_function(wrap_pyfunction!(
        simulation_functions::gradient_linear_gradient_2d,
        &gradient_module
    )?)?;
    gradient_module.add_function(wrap_pyfunction!(
        simulation_functions::gradient_linear_gradient_3d,
        &gradient_module
    )?)?;

    // add simulation::instrument submodule functions
    instrument_module.add_function(wrap_pyfunction!(
        simulation_functions::instrument_gaussian_irf_1d,
        &instrument_module
    )?)?;

    // add simulation::noise submodule functions
    noise_module.add_function(wrap_pyfunction!(
        simulation_functions::noise_poisson_noise_1d,
        &noise_module
    )?)?;
    noise_module.add_function(wrap_pyfunction!(
        simulation_functions::noise_poisson_noise_1d_mut,
        &noise_module
    )?)?;
    noise_module.add_function(wrap_pyfunction!(
        simulation_functions::noise_poisson_noise_3d,
        &noise_module
    )?)?;
    noise_module.add_function(wrap_pyfunction!(
        simulation_functions::noise_poisson_noise_3d_mut,
        &noise_module
    )?)?;

    // attach simulation submodules before attaching to the parent module
    simulation_module.add_submodule(&decay_module)?;
    simulation_module.add_submodule(&gradient_module)?;
    simulation_module.add_submodule(&instrument_module)?;
    simulation_module.add_submodule(&noise_module)?;
    parent_module.add_submodule(&simulation_module)
}
