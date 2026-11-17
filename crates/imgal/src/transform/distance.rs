//! Distance transform functions.
//!
//! TODO

use ndarray::{Array, ArrayBase, ArrayView1, ArrayViewMut1, AsArray, Axis, Dimension, ViewRepr};

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
pub fn squared_edt<'a, T, A, D>(data: A, zero_border: bool) -> Array<f64, D>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let mut res = Array::<f32, D>::zeros(data.raw_dim());
    // do pass 1, extract squared 1D EDT from axis 0
    res.lanes_mut(Axis(0))
        .into_iter()
        .zip(data.lanes(Axis(0)).into_iter())
        .for_each(|(mut r, d)| {
            // TODO pass anisotropy correctly...might not need &mut here.
            left_right_sweep(d, &mut r, 1.0, zero_border);
        });
    todo!();
}

/// This function performs the initial "left/right" or "forward/backward" sweep
/// along the data.
fn left_right_sweep<T>(
    input: ArrayView1<T>,
    output: &mut ArrayViewMut1<f32>,
    aniso_val: f32,
    zero_border: bool,
) where
    T: AsNumeric,
{
    let n = input.len();
    // TODO return early or error here?
    if n == 0 {
        return;
    }
    let zero = T::default();
    let mut temp = input[0];
    output[0] = if temp == zero {
        0.0
    } else if zero_border {
        aniso_val
    } else {
        f32::INFINITY
    };
    input.iter().enumerate().for_each(|(i, &v)|{
        if v == zero {
            output[i] = 0.0;
        } else if v == temp {
            output[i] = output[i - 1] + aniso_val;
        } else {
            output[i] = aniso_val;
            output[i - 1] = if input[i - 1] != zero {
                aniso_val
            } else {
                0.0
            };
            temp = input[i];
        }
    });
    if zero_border {
        output[n - 1] = if input[n - 1] != zero {
            aniso_val
        } else {
            0.0
        };
    }
    todo!();
}

/// Finds the lower envolope of the data.
fn find_lower_envelope() {
    todo!();
}
