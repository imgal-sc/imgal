use std::cmp::Ordering;

use ndarray::{Array, ArrayBase, ArrayD, ArrayView1, AsArray, Axis, Dimension, IxDyn, ViewRepr};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Compute the linear percentile over an n-dimensional array.
///
/// # Description
///
/// Calculates percentiles using linear interpolation between data points. The
/// computation can be performed either on the entire array (flattened) or along
/// a specified axis. The linear percentile is computed using:
///
/// ```text
/// h = (n - 1) × p
/// j = ⌊h⌋
/// γ = h - j
/// percentile = (1 - γ) × v[j] + γ × v[j+1]
/// ```
///
/// Where:
/// - `n` is the array length.
/// - `p` is the percentile in range `0` to `100`.
/// - `v[j]` is the value at index `j`.
/// - `⌊h⌋` is the floor function of `h`.
///
/// When `γ` is close to zero (within `epsilon`), the result is simply `v[j]`,
/// avoiding unnecessary interpolation.
///
/// # Arguments
///
/// * `data`: An n-dimensional image or array.
/// * `percentile`: The percentile value in thae range `0.0` to `100.0`. Values
///   out side this range will be clamped.
/// * `axis`: The axis to compute percentiles along. If `None`, the input `data`
///   is flattened and a single percentile value is returned.
/// * `epsilon`: The tolerance value used to decide the if the fractional index
///   is an integer. If `None`, then `epsilon = 1e-12`.
///
/// # Returns
///
/// * `Ok(ArrayD<f64>)`: The linear percentile of the input data. If `axis` is
///   `None`, the result shape is `(1,)` and contains a single percentile value
///   of the flattened input `data`. If `axis` is a valid axis value, the
///   result has the same shape as `data` with `axis` removed and contains the
///   percentiles calculated along `axis`.
/// * `Err(ImgalError)`: If `axis >= data.ndim()`.
pub fn linear_percentile<'a, T, A, D>(
    data: A,
    percentile: f64,
    axis: Option<usize>,
    epsilon: Option<f64>,
) -> Result<ArrayD<f64>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();

    // validate the input data, no empty arrays
    if view.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" });
    }

    let per_arr = match axis {
        Some(ax) => {
            // validate the axis value
            if ax >= view.ndim() {
                return Err(ImgalError::InvalidAxis {
                    axis_idx: ax,
                    dim_len: view.ndim(),
                });
            }
            let mut shape = view.shape().to_vec();
            shape.remove(ax);
            let mut arr = ArrayD::<f64>::zeros(IxDyn(&shape));
            // compute the percentile for each 1D lane along "axis"
            let lanes = view.lanes(Axis(ax));
            lanes.into_iter().zip(arr.iter_mut()).for_each(|(ln, pr)| {
                *pr = linear_percentile_1d(ln, percentile, epsilon);
            });
            arr
        }
        None => {
            // flatten the input array and compute the percentile
            let val_arr = view.to_owned().into_flat();
            let per = linear_percentile_1d(val_arr.view(), percentile, epsilon);
            Array::from_vec(vec![per]).into_dyn()
        }
    };

    Ok(per_arr)
}

/// 1-dimensional linear percentile.
fn linear_percentile_1d<T>(data: ArrayView1<T>, percentile: f64, epsilon: Option<f64>) -> f64
where
    T: AsNumeric,
{
    let epsilon = epsilon.unwrap_or(1e-12);

    // compute the percentile value using linear interpolation instead of
    // sorting the value array, get the "j" element via unstable selection if
    // "h" is an integer with epsilon value, return the percentile value
    let p_clamp = percentile.clamp(0.0, 100.0);
    let mut val_arr = data.to_vec();
    let p = p_clamp / 100.0;
    let h = (val_arr.len() as f64 - 1.0) * p;
    let j = h.floor() as usize;
    let gamma = h - j as f64;
    if gamma.abs() < epsilon {
        val_arr.select_nth_unstable_by(j, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
        return val_arr[j].to_f64();
    }
    val_arr.select_nth_unstable_by(j, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
    let v_j = val_arr[j].to_f64();
    val_arr.select_nth_unstable_by(j + 1, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
    let v_j1 = val_arr[j + 1].to_f64();

    (1.0 - gamma) * v_j + gamma * v_j1
}
