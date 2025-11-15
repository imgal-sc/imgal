use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

use crate::traits::numeric::AsNumeric;

/// Compute the linear percentile for a given array.
///
/// # Description
///
/// blah blah
///
/// # Arguments
///
/// * `data`:
///
/// # Returns
///
/// * `f64`: The value at the given percentile from the data
pub fn linear_percentile<'a, T, A, D>(data: A, p: T, epsilon: Option<f64>) -> f64
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    // set optional parameters if needed
    let epsilon = epsilon.unwrap_or(1e-12);

    // flatten input array into 1D and sort
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let mut val_arr = view.to_owned().into_flat().to_vec();
    val_arr.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // clamp input parameter "p" to 0..100 range
    let mut p_clamp = p.to_f64();
    if p_clamp < 0.0 {
        p_clamp = 0.0;
    } else if p_clamp > 100.0 {
        p_clamp = 100.0;
    }

    // return early for edge cases 0 and 100th percentiles
    let dl = val_arr.len();
    if p_clamp == 0.0 {
        return val_arr[0].to_f64();
    }
    if p_clamp == 100.0 {
        return val_arr[dl - 1].to_f64();
    }

    // compute the percentile value using linear interpolation
    // if "h" is an integer with epsilon value, return the percentile value
    let p = p_clamp / 100.0;
    let h = (dl as f64 - 1.0) * p;
    let j = h.floor() as usize;
    let gamma = h - j as f64;
    if gamma.abs() < epsilon {
        return val_arr[j].to_f64();
    }
    let v_j = val_arr[j].to_f64();
    let v_j1 = val_arr[j + 1].to_f64();

    (1.0 - gamma) * v_j + gamma * v_j1
}
