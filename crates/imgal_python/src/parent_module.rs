use pyo3::prelude::*;

/// TODO
#[pymodule(name = "imgal")]
mod imgal_module {
    #[pymodule_export]
    use crate::child_modules::colocalization_module::colocalization;
    #[pymodule_export]
    use crate::child_modules::copy_module::copy;
    #[pymodule_export]
    use crate::child_modules::distribution_module::distribution;
    #[pymodule_export]
    use crate::child_modules::filter_module::filter;
    #[pymodule_export]
    use crate::child_modules::image_module::image;
    #[pymodule_export]
    use crate::child_modules::integration_module::integration;
    #[pymodule_export]
    use crate::child_modules::kernel_module::kernel;
    #[pymodule_export]
    use crate::child_modules::overlay_module::overlay;
    #[pymodule_export]
    use crate::child_modules::parameter_module::parameter;
    #[pymodule_export]
    use crate::child_modules::phasor_module::phasor;
    #[pymodule_export]
    use crate::child_modules::simulation_module::simulation;
    #[pymodule_export]
    use crate::child_modules::spatial_module::spatial;
    #[pymodule_export]
    use crate::child_modules::statistics_module::statistics;
    #[pymodule_export]
    use crate::child_modules::threshold_module::threshold;
    #[pymodule_export]
    use crate::child_modules::transform_module::transform;
}
