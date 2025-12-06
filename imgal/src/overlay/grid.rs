use ndarray::ArrayViewMut2;

use crate::traits::numeric::AsNumeric;

/// Apply a grid over a 2-dimensional image.
///
/// # Description
///
/// Applies an adjustable regular grid on an input 2-dimensional array.
///
/// # Arguments
///
/// * `data`: The input 2-dimensional array.
/// * `spacing`: The distance in pixels between grid lines.
pub fn grid_2d_mut<T>(data: &mut ArrayViewMut2<T>, spacing: usize)
where
    T: AsNumeric,
{
    data.indexed_iter_mut().for_each(|((r, c), v)| {
        if r % spacing == 0 || c % spacing == 0 {
            *v = T::MAX;
        }
    });
}
