use ndarray::{ArrayBase, ArrayView, AsArray, Dimension, ViewRepr};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Create a division tiling image stack.
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// * `data`:
/// * `div`:
/// * `factor`: Must be 1 or larger.
///
/// # Returns
///
/// * `Ok(ArrayView<'a, T, D::Larger>)`:
/// * `Err(ImgalError)`:
pub fn div_tile<'a, T, A, D>(
    data: A,
    div: usize,
    factor: usize,
) -> Result<ArrayView<'a, T, D::Larger>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let div = div * factor;
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let shape = view.shape().to_vec();
    shape
        .iter()
        .enumerate()
        .filter(|&(_, &v)| !v.is_multiple_of(div))
        .try_for_each(|(i, &v)| {
            Err(ImgalError::InvalidAxisValueNotAMultipleOf {
                arr_name: "shape",
                axis_idx: i,
                multiple: v,
            })
        })?;

    // construct start and stop positions for each axis
    let tile_positions: Vec<Vec<(usize, usize)>> = shape
        .iter()
        .map(|&v| get_div_start_stop_positions(div, v))
        .collect();

    todo!("n-dimensional image tiling is still under development.");
}

/// descrip
///
/// # Arguments
///
/// * `div`:
/// * `axis_len`: The length of the axis to compute start and stop positions.
///   This function assumes that `axis_len.is_multiple_of(div) == true`.
///
/// # Returns
///
/// * `Vec<(usize, usize)>`: A tuple of start and stop positions,
///   `(start, stop)` along an axis.
fn get_div_start_stop_positions(div: usize, axis_len: usize) -> Vec<(usize, usize)> {
    let mut start_stop_arr: Vec<(usize, usize)> = Vec::with_capacity(div);
    (0..div).fold(0, |acc, _| {
        let start = acc;
        let stop = acc + (axis_len / div);
        start_stop_arr.push((start, stop));
        stop
    });

    start_stop_arr
}
