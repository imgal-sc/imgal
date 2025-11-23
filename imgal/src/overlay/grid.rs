use ndarray::ArrayViewMut2;

use crate::traits::numeric::AsNumeric;

/// Apply a grid over a 2-dimensional image.
///
/// # Description
///
/// This function applies an adjustable, via the `spacing` parameter, regular
/// grid on the input 2-dimensonal image.
///
/// # Arguments
///
/// * `data`: The 2-dimensonal image.
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
