use ndarray::{ArrayD, ArrayViewD, Slice};
use crate::traits::numeric::AsNumeric;

/// Pad an array isometrically with zeros.
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

    // compute the new shape size
    let src_shape = data.shape();
    let mut dst_shape = vec![0; src_shape.len()];
    src_shape
        .iter()
        .zip(dst_shape.iter_mut())
        .for_each(|(s, d)| {
            *d = s + 2 * pad;
        });

    // create a new array with padded dimensions and create a slice view of
    // the padded array using the original image dimensions, and copy the data
    let mut pad_arr = ArrayD::<T>::default(dst_shape);
    let mut pad_view = pad_arr.view_mut();
    pad_view.slice_each_axis_inplace(|ax| {
        Slice {
            start: pad as isize,
            end: Some((pad + src_shape[ax.axis.index()]) as isize),
            step: 1,
        }
    });
    pad_view.assign(&data);

    pad_arr
}
