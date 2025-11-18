use ndarray::{Array, ArrayBase, ArrayD, ArrayView1, AsArray, Axis, Dimension, IxDyn, ViewRepr};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Compute the linear percentile over an n-dimensonal array
///
/// # Description
///
/// This funtion computes the percentile of an entire array or percentiles along
/// a given axis by creating 1-dimensional views along "axis".
///
/// # Arguments
///
/// * `data`: An n-dimensional image or array.
/// * `p`: The percentile value in the range (0..100).
/// * `axis`: The axis to compute percentiles along. If `None`, the input `data`
///    is flattened and a single percentile value is returned.
/// * `epsilon`: The tolerance value used to decide the if the fractional index
///    is an integer, default = 1e-12.
///
/// # Returns
///
/// * `Ok(ArrayD<f64>)`: The linear percentile of the input data. If `axis` is
///    `None`, the result shape is `(1,)` and contains a single percentile value
///    of the flattened input `data`. If `axis` is a valid axis value, the
///    result has the same shape as `data` with `axis` removed and contains the
///    percentiles calculated along `axis`.
/// * `Err(ImgalError)`: If `axis` is >= the number of dimensions of `data`.
pub fn linear_percentile<'a, T, A, D>(
    data: A,
    p: T,
    axis: Option<usize>,
    epsilon: Option<f64>,
) -> Result<ArrayD<f64>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    // create array view and pattern match on "axis"
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let per_arr = match axis {
        None => {
            let val_arr = view.to_owned().into_flat();
            let per = linear_percentile_1d(val_arr.view(), p, epsilon);
            Array::from_vec(vec![per]).into_dyn()
        }
        Some(ax) => {
            // validate axis
            if ax >= view.ndim() {
                return Err(ImgalError::InvalidAxis {
                    axis_idx: ax,
                    dim_len: view.ndim(),
                });
            }

            // create output array
            let mut shape = view.shape().to_vec();
            shape.remove(ax);
            let mut arr = ArrayD::<f64>::zeros(IxDyn(&shape));

            // compute the percentile for each 1D lane along "axis"
            let lanes = view.lanes(Axis(ax));
            lanes.into_iter().zip(arr.iter_mut()).for_each(|(ln, pr)| {
                *pr = linear_percentile_1d(ln, p, epsilon);
            });

            arr
        }
    };

    Ok(per_arr)
}

/// 1-dimensonal linear percentile.
fn linear_percentile_1d<T>(data: ArrayView1<T>, p: T, epsilon: Option<f64>) -> f64
where
    T: AsNumeric,
{
    // set optional parameters if needed
    let epsilon = epsilon.unwrap_or(1e-12);

    // flatten input array into 1D
    let mut val_arr = data.to_vec();
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
