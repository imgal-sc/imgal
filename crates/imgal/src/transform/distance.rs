//! Distance transform functions.
//!
//! TODO

use ndarray::{
    Array, ArrayBase, ArrayView1, ArrayViewMut1, AsArray, Axis, Dimension, Ix1, ViewRepr,
};

use crate::prelude::*;

/// Squared Euclidean distance transform
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// * `data`:
/// * `anisotropy`:
/// * `zero_border`:
///
/// # Returns
///
/// todo
pub fn squared_edt<'a, T, A, B, D>(data: A, anisotropy: B, zero_border: bool) -> Array<f32, D>
where
    A: AsArray<'a, T, D>,
    B: AsArray<'a, f32, Ix1>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let anisotropy: ArrayBase<ViewRepr<&'a f32>, Ix1> = anisotropy.into();
    let ndim = data.ndim();
    let mut res = Array::<f32, D>::zeros(data.raw_dim());
    // TODO do not allow empty arrays
    // do pass 1, extract squared 1D EDT from axis 0
    res.lanes_mut(Axis(0))
        .into_iter()
        .zip(data.lanes(Axis(0)).into_iter())
        .for_each(|(mut r, d)| {
            initial_edt_sweep(d, &mut r, anisotropy[0], zero_border);
        });
    todo!();
}

/// Perform the initial EDT sweep.
///
/// # Description
///
/// This function performs the initial Euclidean Distance Transform sweep (EDT)
/// along 1D views (*i.e.* lanes). The first pass runs from the left to right
/// and evaluates if the current pixel/voxel is background, the same value as
/// the current working label (`temp`) or a different non-zero label. The second
/// pass uses the results from the first pass and runs in reverse, right to
/// left.
///
/// # Arguments
///
/// * `input`:
/// * `output`:
/// * `aniso_val`:
/// * `zero_border`:
fn initial_edt_sweep<T>(
    input: ArrayView1<T>,
    output: &mut ArrayViewMut1<f32>,
    aniso_val: f32,
    zero_border: bool,
) where
    T: AsNumeric,
{
    let n = input.len();
    let zero = T::default();
    let mut temp = input[0];
    output[0] = if temp == zero {
        0.0
    } else if zero_border {
        aniso_val
    } else {
        f32::INFINITY
    };
    input.iter().skip(1).enumerate().for_each(|(i, &v)| {
        if v == zero {
            output[i] = 0.0;
        } else if v == temp {
            output[i] = output[i - 1] + aniso_val;
        } else {
            output[i] = aniso_val;
            output[i - 1] = if input[i - 1] != zero { aniso_val } else { 0.0 };
            temp = input[i];
        }
    });
    if zero_border {
        output[n - 1] = if input[n - 1] != zero { aniso_val } else { 0.0 };
    }
    let start = if zero_border { 1 } else { 0 };
    (start..n - 1).rev().for_each(|i| {
        output[i] = output[i].min(output[i + 1] + aniso_val);
    });
    output.iter_mut().for_each(|v| *v *= *v);
}

/// Finds the lower envolope of the data.
fn find_lower_envelope() {
    todo!();
}
