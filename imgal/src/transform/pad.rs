use ndarray::{ArrayBase, ArrayD, ArrayViewMutD, AsArray, Axis, Dimension, Slice, ViewRepr};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Pad an n-dimensional array with a constant value.
///
/// # Description
///
/// Pads an n-dimensional array with a constant value symmetrically or
/// asymmetrically, along each axis. Symmetric padding increases each axis
/// length by `2 * pad`, where `pad` is the value specified in `pad_config` for
/// that axis. Asymmetric padding increases each axis length by `pad`, adding
/// the specified number of elements at the end of the axis.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array to be padded.
/// * `value`: The constant value to use for padding.
/// * `pad_config`: A slice specifying the pad width for each axis of `data`.
/// * `direction`: A `u8` value to indicate which direction to pad. There are
///   three valid pad directions:
///    - 0: End (right or bottom)
///    - 1: Start (left or top)
///    - 2: Symmetric (both sides)
///
///   If `None`, then `direction = 2` (symmetric padding).
///
/// # Returns
///
/// * `Ok(ArrayD<T>)`: A new constant value padded array containing the input
///   data.
/// * `Err(ImgalError):` If `pad_config.len() != data.ndim()`.
pub fn constant_pad<'a, T, A, D>(
    data: A,
    value: T,
    pad_config: &[usize],
    direction: Option<u8>,
) -> Result<ArrayD<T>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();

    // validate pad_config length
    let src_shape = view.shape().to_vec();
    let sl = src_shape.len();
    if sl != pad_config.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "shape",
            a_arr_len: sl,
            b_arr_name: "pad_config",
            b_arr_len: pad_config.len(),
        });
    }

    // return a copy of the input data if pad config is all zero
    if pad_config.iter().all(|&v| v == 0) {
        return Ok(view.into_dyn().to_owned());
    }

    // validate pad directions
    let direction = direction.unwrap_or(2);
    if direction > 2 {
        return Err(ImgalError::InvalidParameterValueGreater {
            param_name: "direction",
            value: 2,
        });
    }

    // create a constant value padded array and assign source data to a sliced
    // view of the padded output
    let pad_shape: Vec<usize>;
    match direction {
        0 | 1 => {
            // asymmetrical pad
            pad_shape = create_pad_shape(&src_shape, pad_config, false);
        }
        _ => {
            // symmetrical pad
            pad_shape = create_pad_shape(&src_shape, pad_config, true);
        }
    }
    let mut pad_arr = ArrayD::from_elem(pad_shape, value);
    let mut pad_view = pad_arr.view_mut();
    slice_pad_view(&mut pad_view, &src_shape, pad_config, direction);
    pad_view.assign(&view);

    Ok(pad_arr)
}

/// Pad an n-dimensional array with reflected values.
///
/// # Description
///
/// Pads an n-dimensional array with reflected values symmetrically or
/// asymmetrically, along each axis. Symmetric padding increases each axis
/// length by `2 * pad`, where `pad` is the value specified in `pad_config` for
/// that axis. Asymmetric padding increases each axis length by `pad`, adding
/// the specified number of elements at the end of the axis.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array to be padded.
/// * `pad_config`: A slice specifying the pad width for each axis of `data`.
/// * `direction`: A `u8` value to indicate which direction to pad. There are
///   three valid pad directions:
///    - 0: End (right or bottom)
///    - 1: Start (left or top)
///    - 2: Symmetric (both sides)
///
///   If `None`, then `direction = 2` (symmetric padding).
///
/// # Returns
///
/// * `Ok(ArrayD<T>)`: A new reflected value padded array containing the input
///   data.
/// * `Err(ImgalError):` If `pad_config.len() != data.ndim()`.
pub fn reflect_pad<'a, T, A, D>(
    data: A,
    pad_config: &[usize],
    direction: Option<u8>,
) -> Result<ArrayD<T>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();

    // validate pad_config length
    let src_shape = view.shape().to_vec();
    let sl = src_shape.len();
    if sl != pad_config.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "shape",
            a_arr_len: sl,
            b_arr_name: "pad_config",
            b_arr_len: pad_config.len(),
        });
    }

    // validate pad values are within valid range
    pad_config
        .iter()
        .zip(src_shape.iter())
        .enumerate()
        .filter(|&(_, (&p, &s))| p >= s)
        .try_for_each(|(i, (&_, &s))| {
            return Err(ImgalError::InvalidAxisValueGreaterEqual {
                arr_name: "pad_config",
                axis_idx: i,
                value: s,
            });
        })?;

    // return a copy of the input data if pad config is all zero
    if pad_config.iter().all(|&v| v == 0) {
        return Ok(view.into_dyn().to_owned());
    }

    // validate pad directions
    let direction = direction.unwrap_or(2);
    if direction > 2 {
        return Err(ImgalError::InvalidParameterValueGreater {
            param_name: "direction",
            value: 2,
        });
    }

    // create a zero padded array and reflect data into the pad
    let mut pad_arr = zero_pad(view, pad_config, Some(direction))?;
    pad_config
        .iter()
        .zip(src_shape.iter())
        .enumerate()
        .filter(|&(_, (&p, &_))| p != 0)
        .for_each(|(i, (&p, &s))| {
            let pad_view = pad_arr.view_mut();
            match direction {
                // reflect data into the "end" pad
                0 => {
                    let (src_data, mut end_pad) = pad_view.split_at(Axis(i), s);
                    let mut end_reflect =
                        src_data.slice_axis(Axis(i), Slice::from((s - p - 1)..(s - 1)));
                    end_reflect.invert_axis(Axis(i));
                    end_pad.assign(&end_reflect);
                }
                // reflect data into the "start" pad
                1 => {
                    let (mut start_pad, src_data) = pad_view.split_at(Axis(i), p);
                    let mut start_reflect = src_data.slice_axis(Axis(i), Slice::from(1..p + 1));
                    start_reflect.invert_axis(Axis(i));
                    start_pad.assign(&start_reflect);
                }
                // reflect data symmetrically
                _ => {
                    let (mut start_pad, chunk) = pad_view.split_at(Axis(i), p);
                    let (src_data, mut end_pad) = chunk.split_at(Axis(i), s);
                    let mut start_reflect = src_data.slice_axis(Axis(i), Slice::from(1..p + 1));
                    start_reflect.invert_axis(Axis(i));
                    start_pad.assign(&start_reflect);
                    let mut end_reflect =
                        src_data.slice_axis(Axis(i), Slice::from((s - p - 1)..(s - 1)));
                    end_reflect.invert_axis(Axis(i));
                    end_pad.assign(&end_reflect);
                }
            }
        });

    Ok(pad_arr)
}

/// Pad an n-dimensional array with zeros.
///
/// # Description
///
/// Pads an n-dimensional array with zeros symmetrically or asymmetrically,
/// along each axis. Symmetric padding increases each axis length by `2 * pad`,
/// where `pad` is the value specified in `pad_config` for that axis.
/// Asymmetric padding increases each axis length by `pad`, adding the specified
/// number of elements at the end of the axis.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array to be padded.
/// * `pad_config`: A slice specifying the pad width for each axis of `data`.
/// * `direction`: A `u8` value to indicate which direction to pad. There are
///   three valid pad directions:
///    - 0: End (right or bottom)
///    - 1: Start (left or top)
///    - 2: Symmetric (both sides)
///
///   If `None`, then `direction = 2` (symmetric padding).
///
/// # Returns
///
/// * `Ok(ArrayD<T>)`: A new zero padded array containing the input data.
/// * `Err(ImgalError):` If `pad_config.len() != data.ndim()`.
pub fn zero_pad<'a, T, A, D>(
    data: A,
    pad_config: &[usize],
    direction: Option<u8>,
) -> Result<ArrayD<T>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();

    // validate pad_config length
    let src_shape = view.shape().to_vec();
    let sl = src_shape.len();
    if sl != pad_config.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "shape",
            a_arr_len: sl,
            b_arr_name: "pad_config",
            b_arr_len: pad_config.len(),
        });
    }

    // return a copy of the input data if pad config is all zero
    if pad_config.iter().all(|&v| v == 0) {
        return Ok(view.into_dyn().to_owned());
    }

    // validate pad directions
    let direction = direction.unwrap_or(2);
    if direction > 2 {
        return Err(ImgalError::InvalidParameterValueGreater {
            param_name: "direction",
            value: 2,
        });
    }

    // create a zero padded array and assign source data to a sliced view of the
    // padded output
    let pad_shape: Vec<usize>;
    match direction {
        0 | 1 => {
            // asymmetrical pad
            pad_shape = create_pad_shape(&src_shape, pad_config, false);
        }
        _ => {
            // symmetrical pad
            pad_shape = create_pad_shape(&src_shape, pad_config, true);
        }
    }
    let mut pad_arr = ArrayD::<T>::default(pad_shape);
    let mut pad_view = pad_arr.view_mut();
    slice_pad_view(&mut pad_view, &src_shape, pad_config, direction);
    pad_view.assign(&view);

    Ok(pad_arr)
}

/// Construct a padded shape vector.
///
/// # Arguments
///
/// * `shape`: The input shape to pad.
/// * `pad_config`: A slice specifying the pad width per axis.
/// * `symmetric`: If `true`, each axis increases by `pad * 2`. If `false`, each
///   axis increases by `pad`.
#[inline]
fn create_pad_shape(shape: &[usize], pad_config: &[usize], symmetric: bool) -> Vec<usize> {
    let mut pad_shape = vec![0; shape.len()];
    shape
        .iter()
        .zip(pad_config.iter())
        .zip(pad_shape.iter_mut())
        .for_each(|((&s, &p), d)| {
            if symmetric {
                *d = s + 2 * p;
            } else {
                *d = s + p;
            }
        });

    pad_shape
}

/// Slice a mutable view of a padded array back into its initial shape. This
/// function is used to create a mutable region of the same dimensions as the
/// source data _in_ the new padded array. This specific mutable view is used
/// to copy the original data into the new padded array.
///
/// # Arguments
///
/// * `view`: The mutable ArrayViewD to slice in place.
/// * `slice_shape`: The shape to slice the mutable view into.
/// * `pad_config`: A slice specifying the pad width for each axis of `view`.
/// * `direction`: A `u8` value indicating pad direction. `0` or end padding
///   starts at slice index 0, while `1` and `2` (_i.e._ start and symmetric
///   padding) start at slice index `pad + s` where `s` is the length of the
///   current axis.
#[inline]
fn slice_pad_view<T>(
    view: &mut ArrayViewMutD<T>,
    slice_shape: &[usize],
    pad_config: &[usize],
    direction: u8,
) where
    T: AsNumeric,
{
    // slice the mutable view on axes that have been padded, if the pad value
    // for a given axis is 0, do not slice
    pad_config
        .iter()
        .zip(slice_shape.iter())
        .enumerate()
        .filter(|(_, (p, _))| **p != 0)
        .for_each(|(i, (&p, &s))| {
            let ax_slice: Slice;
            match direction {
                0 => {
                    ax_slice = Slice {
                        start: 0 as isize,
                        end: Some(s as isize),
                        step: 1,
                    }
                }
                _ => {
                    ax_slice = Slice {
                        start: p as isize,
                        end: Some((p + s) as isize),
                        step: 1,
                    }
                }
            }
            view.slice_axis_inplace(Axis(i), ax_slice);
        });
}
