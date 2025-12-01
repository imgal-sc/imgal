use ndarray::{ArrayBase, ArrayD, ArrayViewMutD, AsArray, Axis, Dimension, Slice, ViewRepr};

use crate::traits::numeric::AsNumeric;

/// Pad an n-dimensional array with a constant value.
///
/// # Description
///
/// This function pads an n-dimensional array with constant values at specified
/// axes (asymmetrical padding) or isometrically (symmetrical padding). If
/// padding is asymmetrical then the specified padded axes will increase by
/// `pad`. If padding is symmetrical then each dimension increases by 2 * `pad`.
///
/// # Arguments
///
/// * `data`: An n-dimensional array.
/// * `value`: The constant value to pad with.
/// * `pad`: The number of contant values to pad with. If `axes` is `None` then
///   each axis increases by 2 * `pad`, otherwise each axis specified in `axes`
///   increases by `pad`.
/// * `axes`: An array of axes specifiying which axis to pad with constant
///   values. Each axis in `axes` is extended at the end (_i.e._ "right" side
///   padding) by the pad amount with constant values. If `axes` is `None` then
///   the input array is padded isometrically with the given constant value.
///
/// # Returns
///
/// * `ArrayD<T>`: A new constant value padded array containing the input data.
pub fn constant_pad<'a, T, A, D>(data: A, value: T, pad: usize, axes: Option<&[usize]>) -> ArrayD<T>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    // create an array view of the data
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();

    // return a copy of the input data if pad is 0
    if pad == 0 {
        return view.into_dyn().to_owned();
    }

    // create a new array with padded dimensions and create a slice view of
    // the padded array using the original image dimensions, and copy the data
    let src_shape = view.shape();
    let pad_shape = create_pad_shape(pad, src_shape, axes);
    let mut pad_arr = ArrayD::from_elem(pad_shape, value);
    let mut pad_view = pad_arr.view_mut();
    slice_pad_view(&mut pad_view, pad, src_shape, axes);
    pad_view.assign(&view);

    pad_arr
}

/// Pad an n-dimensional array with reflected values.
///
/// # Description
///
/// This function pads an n-dimensional array with reflected values at specified
/// axes (asymmetrical padding) or isometrically (symmetrical padding). If
/// padding is asymmetrical then the specified padded axes will increase by
/// `pad`. If padding is symmetrical then each dimension increases by 2 * `pad`.
///
/// # Arguments
///
/// * `data`: An n-dimensional array.
/// * `pad`: The number of reflected values to pad with. If `axes` is `None`
///   then each axis increases by 2 * `pad`, otherwise each axis specified in
///   `axes` increases by `pad`.
/// * `axes`: An array of axes specifiying which axis to pad with reflected
///   values. Each axis in `axes` is extended at the end (_i.e._ "right" side
///   padding) by the pad amount with reflected values. If `axes` is `None` then
///   the input array is padded isometrically with reflected values.
///
/// # Returns
///
/// * `ArrayD<T>`: A new reflected value padded array containing the input data.
pub fn reflect_pad<'a, T, A, D>(data: A, pad: usize, axes: Option<&[usize]>) -> ArrayD<T>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    // create an array view of the data
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();

    // return a copy of the input data if pad is 0
    if pad == 0 {
        return view.into_dyn().to_owned();
    }

    // create zero padded array with requested pad configuration and reflect
    let src_shape = view.shape().to_vec();
    let mut pad_arr = zero_pad(view.into_dyn(), pad, axes);
    match axes {
        Some(axes) => {
            // asymmetrical pad shape
            src_shape.iter().enumerate().for_each(|(i, &d)| {
                // only slice requested axes
                if axes.contains(&i) {
                    let pad_view = pad_arr.view_mut();
                    let (src_data, mut end_pad) = pad_view.split_at(Axis(i), d);
                    // reflect data into the "end" pad
                    let mut end_reflect =
                        src_data.slice_axis(Axis(i), Slice::from((d - pad - 1)..(d - 1)));
                    end_reflect.invert_axis(Axis(i));
                    end_pad.assign(&end_reflect);
                }
            })
        }
        None => {
            // symmetrical pad shape
            src_shape.iter().enumerate().for_each(|(i, &d)| {
                let pad_view = pad_arr.view_mut();
                let (mut start_pad, rest) = pad_view.split_at(Axis(i), pad);
                let (src_data, mut end_pad) = rest.split_at(Axis(i), d);
                // reflect data into the "start" pad
                let mut start_reflect = src_data.slice_axis(Axis(i), Slice::from(1..pad + 1));
                start_reflect.invert_axis(Axis(i));
                start_pad.assign(&start_reflect);
                // reflect data into the "end" pad
                let mut end_reflect =
                    src_data.slice_axis(Axis(i), Slice::from((d - pad - 1)..(d - 1)));
                end_reflect.invert_axis(Axis(i));
                end_pad.assign(&end_reflect);
            });
        }
    }

    pad_arr
}

/// Pad an n-dimensional array with zeros.
///
/// # Description
///
/// This function pads an n-dimensional array with zeros at specified axes
/// (asymmetrical padding) or isometrically (symmetrical padding). If padding is
/// asymmetrical then the specified padded axes will increase by `pad`. If
/// padding is symmetrical then each dimension increases by 2 * `pad`.
///
/// # Arguments
///
/// * `data`: An n-dimensional array.
/// * `pad`: The number of zeros to pad with. If `axes` is `None` then each axis
///   increases by 2 * `pad`, otherwise each axis specified in `axes` increases
///   by `pad`.
/// * `axes`: An array of axes specifiying which axis to pad with zeros. Each
///   axis in `axes` is extended at the end (_i.e._ "right" side padding) by the
///   pad amount with zeros. If `axes` is `None` then the input array is padded
///   isometrically with zeros.
///
/// # Returns
///
/// * `ArrayD<T>`: A new zero padded array containing the input data.
pub fn zero_pad<'a, T, A, D>(data: A, pad: usize, axes: Option<&[usize]>) -> ArrayD<T>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    // create an array view of the data
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();

    // return a copy of the input data if pad is 0
    if pad == 0 {
        return view.into_dyn().to_owned();
    }

    // TODO: ensure axes length does not exceed data.shape().len()
    // TODO: ensure axes values themselves are valid
    // TODO: if axes is none, default to all axes -> isometric padding
    // create a new array with padded dimensions and create a slice view of
    // the padded array using the original image dimensions, and copy the data
    let src_shape = view.shape().to_vec();
    let pad_shape = create_pad_shape(pad, &src_shape, axes);
    let mut pad_arr = ArrayD::<T>::default(pad_shape);
    let mut pad_view = pad_arr.view_mut();
    slice_pad_view(&mut pad_view, pad, &src_shape, axes);
    pad_view.assign(&view);

    pad_arr
}

/// Construct a padded shape vector from a given shape slice and pad value.
///
/// # Arguments
///
/// * `pad`: The number of elements to pad by.
/// * `shape`: The input shape to pad.
/// * `axes`: A slice of axes to pad. If None, each dimension will be padded
///    equally.
#[inline]
fn create_pad_shape(pad: usize, shape: &[usize], axes: Option<&[usize]>) -> Vec<usize> {
    // TODO: consider making axes optional here and use deafult logic for isometric pad
    let mut pad_shape = vec![0; shape.len()];
    match axes {
        Some(axes) => {
            // asymmetrical pad shape
            shape
                .iter()
                .zip(pad_shape.iter_mut())
                .enumerate()
                .for_each(|(i, (s, d))| {
                    if axes.contains(&i) {
                        *d = s + pad;
                    } else {
                        *d = *s;
                    }
                });
        }
        None => {
            // symmetrical pad shape
            shape.iter().zip(pad_shape.iter_mut()).for_each(|(s, d)| {
                *d = s + 2 * pad;
            });
        }
    }

    pad_shape
}

/// Slice a mutable view of a padded array back into its initial shape. This
/// function is used to create a mutable region of the same dimensions as the
/// source data _in_ the new padded array. This specific mutable view is used
/// to copy the original data into the new padded array. To optimize this, the
/// original data and padded view must have the same dimensions.
///
/// # Arguments
///
/// * `view`: The mutable ArrayView to inplace slice.
/// * `pad`: The number of elements to pad by.
/// * `slice_shape`: The shape to slice the view into to.
/// * `axes`: A slice of axes to pad. If None, the view will be sliced equally
///   in each dimension.
#[inline]
fn slice_pad_view<T>(
    view: &mut ArrayViewMutD<T>,
    pad: usize,
    slice_shape: &[usize],
    axes: Option<&[usize]>,
) where
    T: AsNumeric,
{
    match axes {
        Some(axes) => {
            // slice the view asymmetrically
            axes.iter().for_each(|&a| {
                let ax_slice = Slice {
                    start: 0 as isize,
                    end: Some(slice_shape[a] as isize),
                    step: 1,
                };
                view.slice_axis_inplace(Axis(a), ax_slice);
            });
        }
        None => {
            // slice the view symmetrically
            view.slice_each_axis_inplace(|ax| Slice {
                start: pad as isize,
                end: Some((pad + slice_shape[ax.axis.index()]) as isize),
                step: 1,
            });
        }
    }
}
