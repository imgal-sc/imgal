use pyo3::prelude::*;

/// Basic image processing functions.
///
/// This module provides *n*-dimensional basic image functions such as
/// histograms and image normalization.
#[pymodule(submodule)]
pub mod image {
    #[pymodule_export]
    use crate::functions::image_functions::image_histogram;
    #[pymodule_export]
    use crate::functions::image_functions::image_histogram_bin_midpoint;
    #[pymodule_export]
    use crate::functions::image_functions::image_histogram_bin_range;
    #[pymodule_export]
    use crate::functions::image_functions::normalize_percentile_normalize;
}
