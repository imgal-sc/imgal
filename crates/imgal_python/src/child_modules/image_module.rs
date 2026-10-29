use pyo3::prelude::*;

/// Basic image processing functions.
///
/// This module provides *n*-dimensional basic image functions such as
/// histograms and image normalization.
#[pymodule(submodule)]
pub mod image {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::functions::image_functions::image_histogram;
    #[pymodule_export]
    use crate::functions::image_functions::image_histogram_bin_midpoint;
    #[pymodule_export]
    use crate::functions::image_functions::image_histogram_bin_range;
    #[pymodule_export]
    use crate::functions::image_functions::normalize_percentile_normalize;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        crate::utils::register_in_sys_modules(m, "imgal.image")
    }
}
