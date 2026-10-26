use pyo3::prelude::*;

/// Sorting, correlation and other image statistics.
///
/// This module provides various image related statistics
/// (*e.g.* min, max, etc...), sorting and correlation functions.
#[pymodule(submodule)]
pub mod statistics {
    #[pymodule_export]
    use crate::functions::statistics_functions::statistics_effective_sample_size;
    #[pymodule_export]
    use crate::functions::statistics_functions::statistics_kahan_sum;
    #[pymodule_export]
    use crate::functions::statistics_functions::statistics_linear_percentile;
    #[pymodule_export]
    use crate::functions::statistics_functions::statistics_max;
    #[pymodule_export]
    use crate::functions::statistics_functions::statistics_min;
    #[pymodule_export]
    use crate::functions::statistics_functions::statistics_min_max;
    #[pymodule_export]
    use crate::functions::statistics_functions::statistics_pearson;
    #[pymodule_export]
    use crate::functions::statistics_functions::statistics_sum;
    #[pymodule_export]
    use crate::functions::statistics_functions::statistics_weighted_kendall_tau_b;
    #[pymodule_export]
    use crate::functions::statistics_functions::statistics_weighted_merge_sort_mut;
}
