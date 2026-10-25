use pyo3::prelude::*;

/// Phasor compute, calibration, and plot functions.
///
/// This module provides time domain phasor calculation, calibration and
/// plotting helper functions.
#[pymodule(submodule)]
pub mod phasor {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::child_modules::phasor_module::calibration;
    #[pymodule_export]
    use crate::child_modules::phasor_module::plot;
    #[pymodule_export]
    use crate::child_modules::phasor_module::time_domain;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.phasor")
    }
}

/// Phasor calibration functions.
///
/// This module provides time domain phasor calibration functions.
#[pymodule(submodule)]
pub mod calibration {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::phasor_functions::calibration_calibrate_coords;
    #[pymodule_export]
    use crate::functions::phasor_functions::calibration_calibrate_gs_image;
    #[pymodule_export]
    use crate::functions::phasor_functions::calibration_calibrate_gs_image_mut;
    #[pymodule_export]
    use crate::functions::phasor_functions::calibration_calibrate_gs_roi_mut;
    #[pymodule_export]
    use crate::functions::phasor_functions::calibration_modulation_and_phase;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.phasor.calibration")
    }
}

/// Phasor plot related functions.
///
/// This module provides time domain phasor plotting helper functions.
#[pymodule(submodule)]
pub mod plot {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::phasor_functions::plot_gs_mask;
    #[pymodule_export]
    use crate::functions::phasor_functions::plot_gs_modulation;
    #[pymodule_export]
    use crate::functions::phasor_functions::plot_gs_phase;
    #[pymodule_export]
    use crate::functions::phasor_functions::plot_monoexponential_coords;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.phasor.plot")
    }
}

/// Phasor time domain calculation functions.
///
/// This module provides time domain phasor calculation functions.
#[pymodule(submodule)]
pub mod time_domain {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::phasor_functions::time_domain_gs_image;
    #[pymodule_export]
    use crate::functions::phasor_functions::time_domain_gs_roi;
    #[pymodule_export]
    use crate::functions::phasor_functions::time_domain_imaginary_coord;
    #[pymodule_export]
    use crate::functions::phasor_functions::time_domain_real_coord;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.phasor.time_domain")
    }
}
