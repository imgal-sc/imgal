use pyo3::prelude::*;

/// Colocalization analysis on 2D and 3D data.
///
/// This module provides colocalization analysis functions for both 2D and 3D
/// image data.
#[pymodule(submodule)]
pub mod colocalization {
    #[pymodule_export]
    use crate::functions::colocalization_functions::colocalization_pearson_roi_coloc;
    #[pymodule_export]
    use crate::functions::colocalization_functions::colocalization_saca_2d;
    #[pymodule_export]
    use crate::functions::colocalization_functions::colocalization_saca_3d;
    #[pymodule_export]
    use crate::functions::colocalization_functions::colocalization_saca_significance_mask;
}
