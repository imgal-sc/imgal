use core::f64;

use numpy::PyReadwriteArray2;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use imgal::overlay;

/// Apply a grid over a 2-dimensional image.
///
/// Applies an adjustable regular grid on an input 2-dimensional array.
///
/// Args:
///     data: The input 2-dimensional array.
///     spacing: The distance in pixels between grid lines.
#[pyfunction]
#[pyo3(name = "grid_2d_mut")]
pub fn grid_grid_2d_mut<'py>(data: Bound<'py, PyAny>, spacing: usize) -> PyResult<()> {
    if let Ok(mut arr) = data.extract::<PyReadwriteArray2<u8>>() {
        overlay::grid::grid_2d_mut(&mut arr.as_array_mut(), spacing);
        Ok(())
    } else if let Ok(mut arr) = data.extract::<PyReadwriteArray2<u16>>() {
        overlay::grid::grid_2d_mut(&mut arr.as_array_mut(), spacing);
        Ok(())
    } else if let Ok(mut arr) = data.extract::<PyReadwriteArray2<u64>>() {
        overlay::grid::grid_2d_mut(&mut arr.as_array_mut(), spacing);
        Ok(())
    } else if let Ok(mut arr) = data.extract::<PyReadwriteArray2<f32>>() {
        overlay::grid::grid_2d_mut(&mut arr.as_array_mut(), spacing);
        Ok(())
    } else if let Ok(mut arr) = data.extract::<PyReadwriteArray2<f64>>() {
        overlay::grid::grid_2d_mut(&mut arr.as_array_mut(), spacing);
        Ok(())
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}
