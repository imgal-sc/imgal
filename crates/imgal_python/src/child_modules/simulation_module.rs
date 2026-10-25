use pyo3::prelude::*;

/// Decay, instrument, and noise simulation functions.
///
/// This module provides various functions for simulating exponential decays,
/// blobs, instrument response functions (IRF), gradients, random numbers and
/// noise.
#[pymodule(submodule)]
pub mod simulation {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::child_modules::simulation_module::blob;
    #[pymodule_export]
    use crate::child_modules::simulation_module::decay;
    #[pymodule_export]
    use crate::child_modules::simulation_module::gradient;
    #[pymodule_export]
    use crate::child_modules::simulation_module::instrument;
    #[pymodule_export]
    use crate::child_modules::simulation_module::noise;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.simulation")
    }
}

/// Blob simulation functions.
///
/// This module provides functions for simulating *n*-dimensional blobs.
#[pymodule(submodule)]
pub mod blob {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::simulation_functions::blob_gaussian_metaballs;
    #[pymodule_export]
    use crate::functions::simulation_functions::blob_logistic_metaballs;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.simulation.blob")
    }
}

/// Decay simulation functions.
///
/// This module provides functions for simulating 1D and 3D monexponential or
/// multiexponential decay data.
#[pymodule(submodule)]
pub mod decay {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::simulation_functions::decay_gaussian_exponential_decay_1d;
    #[pymodule_export]
    use crate::functions::simulation_functions::decay_gaussian_exponential_decay_3d;
    #[pymodule_export]
    use crate::functions::simulation_functions::decay_ideal_exponential_decay_1d;
    #[pymodule_export]
    use crate::functions::simulation_functions::decay_ideal_exponential_decay_3d;
    #[pymodule_export]
    use crate::functions::simulation_functions::decay_irf_exponential_decay_1d;
    #[pymodule_export]
    use crate::functions::simulation_functions::decay_irf_exponential_decay_3d;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.simulation.decay")
    }
}

/// Image gradient simulation functions.
///
/// This module provides functions for simulating 2D and 3D gradient images.
#[pymodule(submodule)]
pub mod gradient {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::simulation_functions::gradient_linear_gradient_2d;
    #[pymodule_export]
    use crate::functions::simulation_functions::gradient_linear_gradient_3d;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.simulation.gradient")
    }
}

/// Instrument simulation functions.
///
/// This module provides instrument related simulation functions (*e.g* an
/// instrument response function (IRF)).
#[pymodule(submodule)]
pub mod instrument {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::simulation_functions::instrument_gaussian_irf_1d;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.simulation.instrument")
    }
}

/// Noise simulation functions.
///
/// This module provides noise simulation functions for creating new or mutating
/// data with reproducible noise.
#[pymodule(submodule)]
pub mod noise {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::simulation_functions::noise_poisson_noise;
    #[pymodule_export]
    use crate::functions::simulation_functions::noise_poisson_noise_mut;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.simulation.noise")
    }
}
