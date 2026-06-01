use ndarray::{
    Array, ArrayBase, ArrayView, ArrayViewMut, AsArray, Axis, Dimension, RemoveAxis, ViewRepr, Zip,
};
use rayon::prelude::*;

use crate::prelude::*;
use crate::statistics::linear_percentile;

/// Normalize an n-dimensional image using percentile-based minimum and maximum.
///
/// # Description
///
/// Performs percentile-based normalization of an input n-dimensional image with
/// minimum and maximum percentage within the range of `0.0` to `100.0`.
///
/// The normalization is computed as:
///
/// ```text
/// y = (x - min) / (max - min + ε)
/// ```
///
/// Where:
/// - `y` is the normalized output.
/// - `x` is the input.
/// - `min` is the value at the minimum percentile.
/// - `max` is the value at the maximum percentile.
/// - `ε` is a small epsilon value to prevent division by zero.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image to normalize.
/// * `min`: The minimum normalization percentile in the range `0.0` to `100.0`.
/// * `max`: The maximum normalization percentile in the range `0.0` to `100.0`.
/// * `clip`: Boolean to indicate whether to clamp the normalized values to the
///   range `0.0` to `100.0`.
/// * `axis`: The axis to comppute percentiles idependently along. Each subview
///   along this axis normalized with the independent percentiles. If `None`,
///   then the input `data` is flattened.
/// * `epsilon`: A small positive value to avoid division by zero. If `None`,
///   then `epsilon = 1e-20`.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Ok(ArrayD<f64>)`: The percentile normalized n-dimensional image.
/// * `Err(ImgalError)`: If `min` and/or `max` are outside of range `0.0` to
///   `1.0`.
pub fn percentile_normalize<'a, T, A, D>(
    data: A,
    min: f64,
    max: f64,
    clip: bool,
    axis: Option<usize>,
    epsilon: Option<f64>,
    threads: Option<usize>,
) -> Result<Array<f64, D>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension + RemoveAxis,
    T: 'a + AsNumeric,
{
    if !(0.0..=100.0).contains(&min) {
        return Err(ImgalError::InvalidParameterValueOutsideRange {
            param_name: "min",
            value: min,
            min: 0.0,
            max: 100.0,
        });
    }
    if !(0.0..=100.0).contains(&max) {
        return Err(ImgalError::InvalidParameterValueOutsideRange {
            param_name: "max",
            value: max,
            min: 0.0,
            max: 100.0,
        });
    }
    if min > max {
        return Err(ImgalError::InvalidParameterGreater {
            a_param_name: "min",
            b_param_name: "max",
        });
    }
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let epsilon = epsilon.unwrap_or(1e-20);
    match axis {
        Some(ax) => {
            if ax >= data.ndim() {
                return Err(ImgalError::InvalidAxis {
                    axis_idx: ax,
                    dim_len: data.ndim(),
                });
            }
            let ax = Axis(ax);
            let mm: Vec<(f64, f64)> = data.axis_iter(ax).try_fold(Vec::new(), |mut acc, s| {
                let pmin = linear_percentile(&s, min, None, None, None)?[0];
                let pmax = linear_percentile(&s, max, None, None, None)?[0];
                acc.push((pmin, pmax));
                Ok(acc)
            })?;
            let mut norm_arr = Array::from_elem(data.dim(), 0.0);
            let norm_calc =
                |i: usize, a: ArrayView<T, D::Smaller>, mut b: ArrayViewMut<f64, D::Smaller>| {
                    let (pmin, pmax) = mm[i];
                    let denom = pmax - pmin + epsilon;
                    Zip::from(a).and(b.view_mut()).for_each(|&v, n| {
                        let norm = (v.to_f64() - pmin) / denom;
                        *n = if clip { norm.clamp(0.0, 1.0) } else { norm };
                    });
                };
            par!(threads,
                seq_exp: data.axis_iter(ax).zip(norm_arr.axis_iter_mut(ax))
                    .enumerate()
                    .for_each(|(i, (a, b))| norm_calc(i, a, b)),
                par_exp: data.axis_iter(ax).zip(norm_arr.axis_iter_mut(ax))
                    .enumerate()
                    .par_bridge()
                    .for_each(|(i, (a, b))| norm_calc(i, a, b)));
            return Ok(norm_arr);
        }
        None => {
            let pmin = linear_percentile(&data, min, None, None, None)?[0];
            let pmax = linear_percentile(&data, max, None, None, None)?[0];
            let denom = pmax - pmin + epsilon;
            let mut norm_arr = Array::from_elem(data.dim(), 0.0);
            let norm_calc = |v: &T, n: &mut f64| {
                let norm = (v.to_f64() - pmin) / denom;
                *n = if clip { norm.clamp(0.0, 1.0) } else { norm };
            };
            par!(threads,
                seq_exp: Zip::from(data).and(norm_arr.view_mut())
                    .for_each(&norm_calc),
                par_exp: Zip::from(data).and(norm_arr.view_mut())
                    .par_for_each(&norm_calc));
            return Ok(norm_arr);
        }
    }
}
