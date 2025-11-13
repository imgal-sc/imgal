use crate::traits::numeric::AsNumeric;
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Slice};

/// Pad an n-dimensional array isometrically with a constant value.
///
/// # Description
///
/// This function pads an n-dimensional array isometrically (_i.e_ equally on
/// all sides) with a constant value. The output padded array will be larger in
/// each dimension by 2 * `pad`. The input data is centered within the new
/// padded array.
///
/// # Arguments
///
/// * `data`: An n-dimensional array.
/// * `value`: The constant value to pad with.
/// * `pad`: The number of constant values to pad on each side of every axis.
///    Each axis increases by 2 * `pad`.
///
/// # Returns
///
/// * `ArrayD<T>`: A new array containing the input data centered in a
///    constant value padded array.
pub fn isometric_constant<T>(data: ArrayViewD<T>, value: T, pad: usize) -> ArrayD<T>
where
    T: AsNumeric,
{
    // return a copy of the input data if pad is 0
    if pad == 0 {
        return data.to_owned();
    }

    // create a new array with padded dimensions and create a slice view of
    // the padded array using the original image dimensions, and copy the data
    let shape = data.shape();
    let pad_shape = create_pad_shape(pad, shape);
    let mut pad_arr = ArrayD::from_elem(pad_shape, value);
    let mut pad_view = pad_arr.view_mut();
    slice_pad_view(&mut pad_view, pad, shape);
    pad_view.assign(&data);

    pad_arr
}

/// Pad an n-dimensional array isometrically with zeros.
///
/// # Description
///
/// This function pads an n-dimensional array isometrically (_i.e_ equally on
/// all sides) with zero. The output padded array will be larger in each
/// dimension by 2 * `pad`. The input data is centered within the new padded
/// array.
///
/// # Arguments
///
/// * `data`: An n-dimensional array.
/// * `pad`: The number of zeros to pad on each side of every axis. Each axis
///    increases by 2 * `pad`.
///
/// # Returns
///
/// * `ArrayD<T>`: A new array containing the input data centered in a
///    zero-padded array.
pub fn isometric_zero<T>(data: ArrayViewD<T>, pad: usize) -> ArrayD<T>
where
    T: AsNumeric,
{
    // return a copy of the input data if pad is 0
    if pad == 0 {
        return data.to_owned();
    }

    // create a new array with padded dimensions and create a slice view of
    // the padded array using the original image dimensions, and copy the data
    let shape = data.shape();
    let pad_shape = create_pad_shape(pad, shape);
    let mut pad_arr = ArrayD::<T>::default(pad_shape);
    let mut pad_view = pad_arr.view_mut();
    slice_pad_view(&mut pad_view, pad, shape);
    pad_view.assign(&data);

    pad_arr
}

/// Construct a padded shape vector from a given shape slice and pad value.
#[inline]
fn create_pad_shape(pad: usize, shape: &[usize]) -> Vec<usize> {
    let mut p_shape = vec![0; shape.len()];
    shape.iter().zip(p_shape.iter_mut()).for_each(|(s, d)| {
        *d = s + 2 * pad;
    });

    p_shape
}

/// Slice a mutable view of a padded back into its initial shape. This function
/// is used to create a mutable region of the same dimensions as the source data
/// _in_ the new padded array. This specific mutable view is used to copy the
/// original data into the new padded array. To optimize this, the original data
/// and padded view must have the same dimensions.
#[inline]
fn slice_pad_view<T>(view: &mut ArrayViewMutD<T>, pad: usize, shape: &[usize])
where
    T: AsNumeric,
{
    view.slice_each_axis_inplace(|ax| Slice {
        start: pad as isize,
        end: Some((pad + shape[ax.axis.index()]) as isize),
        step: 1,
    });
}
