//! Distance transform functions.
//!
//! TODO

use ndarray::{
    Array, ArrayBase, ArrayView1, ArrayViewMut1, AsArray, Axis, Dimension, Ix1, ViewRepr,
};
use rustfft::num_complex::ComplexFloat;

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
    // now do pass 2 on remaining axes
    if ndim > 1 {
        (1..ndim).for_each(|i| {
            let aniso_val = anisotropy[i];
            res.lanes_mut(Axis(i))
                .into_iter()
                .zip(data.lanes(Axis(i)).into_iter())
                .for_each(|(mut r, d)| {
                    find_lower_envelope(d, &mut r, anisotropy[i], zero_border);
                });
        });
        // return the result for all 2D+ arrays
    } else {
        // return the result for 1D arrays
    }
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
fn find_lower_envelope<T>(
    input: ArrayView1<T>,
    output: &mut ArrayViewMut1<f32>,
    aniso_val: f32,
    zero_border: bool,
) where
    T: AsNumeric,
{
    let n = input.len();
    let zero = T::default();
    let aniso_sq = (aniso_val as f64) * (aniso_val as f64);
    let mut bounds: Vec<(f64, f64)> = Vec::new();
    let mut bounds_inds: Vec<usize> = Vec::new();
    let mut mins: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < n {
        if input[i] == zero {
            i += 1;
            continue;
        }
        // here we find the start and end bounds of the working label, temp
        let start = i;
        let temp = input[i];
        while i < n && input[i] == temp {
            i += 1;
        }
        let end = i;
        bounds.clear();
        if start > 0 || zero_border {
            let s = (start - 1) as f64;
            bounds.push((s, 0.0));
        }
        (start..end)
            .filter(|&j| output[j].is_finite())
            .for_each(|j| bounds.push((j as f64, output[j] as f64)));
        if end < n || zero_border {
            bounds.push((end as f64, 0.0));
        }
        if bounds.is_empty() {
            continue;
        }
        // begin finding the lower envelop using Felzenszwalb & Huttenlocher approach
        bounds_inds.clear();
        mins.clear();
        bounds_inds.push(0);
        mins.push(f64::NEG_INFINITY);
        (1..bounds.len()).for_each(|j| {
            let (q, fq) = bounds[j];
            let s = loop {
                let k = bounds_inds.len() - 1;
                let (r, fr) = bounds[bounds_inds[k]];
                let s = ((fq + aniso_sq * q * q) - (fr + aniso_sq * r * r))
                    / (2.0 * aniso_sq * (q - r));
                if s <= mins[k] {
                    bounds_inds.pop();
                    mins.pop();
                } else {
                    break s;
                }
            };
            bounds_inds.push(j);
            mins.push(s);
        });
        let mut k = 0;
        (start..end).for_each(|j| {
            let x = j as f64;
            while k + 1 < bounds_inds.len() && mins[k + 1] < x {
                k += 1;
            }
            let (q, f) = bounds[bounds_inds[k]];
            output[j] = (f + aniso_sq * (x - q) * (x - q)) as f32;
        });
    }
}
