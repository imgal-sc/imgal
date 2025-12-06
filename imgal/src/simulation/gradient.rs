use ndarray::{Array2, Array3};

/// Create a 2-dimensional array with a linear gradient.
///
/// # Description
///
/// Creates a linear gradient of increasing values from the top of the array to
/// the bottom along the row axis. Setting the `offset` parameter controls how
/// far the gradient extends while the `scale` parameter controls the rate
/// values increase per row.
///
/// # Arguments
///
/// * `offset`: The number of rows from the top of the array that remain at
///   zero.
/// * `scale`: The rate of increase per row. This value controls the steepness
///   of the gradient.
/// * `shape`: The row and col shape of the gradient array.
///
/// # Returns
///
/// * `Array2<f64>`: The 2-dimensional gradient array.
pub fn linear_gradient_2d(offset: usize, scale: f64, shape: (usize, usize)) -> Array2<f64> {
    Array2::from_shape_fn(shape, |(r, _)| {
        if r < offset {
            0.0
        } else {
            (r - offset) as f64 * scale
        }
    })
}

/// Create a 3-dimensional array with a linear gradient.
///
/// # Description
///
/// Creates a linear gradient of increasing values from the top of the array to
/// the bottom along the pln or z axis. Setting the `offset` parameter controls
/// how far the gradient extends while the `scale` parameter controls the rate
/// values increase per pln.
///
/// # Arguments
///
/// * `offset`: The number of plns from the top of the array tha tremain at
///   zero.
/// * `scale`: The rate of increase per pln. This value controls the steepness
///   of the gradient.
/// * `shape`: The pln, row and col shape of the gradient array.
///
/// # Returns
///
/// * `Array3<f64>`: The 3-dimensional gradient array.
pub fn linear_gradient_3d(offset: usize, scale: f64, shape: (usize, usize, usize)) -> Array3<f64> {
    Array3::from_shape_fn(shape, |(p, _, _)| {
        if p < offset {
            0.0
        } else {
            (p - offset) as f64 * scale
        }
    })
}
