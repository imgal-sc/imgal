use ndarray::{Array, ArrayBase, AsArray, Axis, Dimension, RemoveAxis, ViewRepr, Zip};
use rayon::prelude::*;

use crate::error::ImgalError;
use crate::statistics::linear_percentile;
use crate::traits::numeric::AsNumeric;

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
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
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
    parallel: bool,
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
                let pmin = linear_percentile(&s, min, None, None, false)?[0];
                let pmax = linear_percentile(&s, max, None, None, false)?[1];
                acc.push((pmin, pmax));
                Ok(acc)
            })?;
            let mut norm_arr = Array::from_elem(data.dim(), 0.0);
            if parallel {
                data.axis_iter(ax)
                    .zip(norm_arr.axis_iter_mut(ax))
                    .enumerate()
                    .par_bridge()
                    .for_each(|(i, (a, mut b))| {
                        let (pmin, pmax) = mm[i];
                        let denom = pmax - pmin + epsilon;
                        Zip::from(a).and(b.view_mut()).for_each(|&v, n| {
                            let norm = (v.to_f64() - pmin) / denom;
                            *n = if clip { norm.clamp(0.0, 1.0) } else { norm };
                        })
                    });
            } else {
                data.axis_iter(ax)
                    .zip(norm_arr.axis_iter_mut(ax))
                    .enumerate()
                    .for_each(|(i, (a, mut b))| {
                        let (pmin, pmax) = mm[i];
                        let denom = pmax - pmin + epsilon;
                        Zip::from(a).and(b.view_mut()).for_each(|&v, n| {
                            let norm = (v.to_f64() - pmin) / denom;
                            *n = if clip { norm.clamp(0.0, 1.0) } else { norm };
                        })
                    });
            }
            return Ok(norm_arr);
        }
        None => {
            let pmin = linear_percentile(&data, min, None, None, false)?[0];
            let pmax = linear_percentile(&data, max, None, None, false)?[0];
            let denom = pmax - pmin + epsilon;
            let mut norm_arr = Array::from_elem(data.dim(), 0.0);
            if parallel {
                Zip::from(data)
                    .and(norm_arr.view_mut())
                    .par_for_each(|v, n| {
                        let norm = (v.to_f64() - pmin) / denom;
                        *n = if clip { norm.clamp(0.0, 1.0) } else { norm };
                    });
            } else {
                Zip::from(data).and(norm_arr.view_mut()).for_each(|v, n| {
                    let norm = (v.to_f64() - pmin) / denom;
                    *n = if clip { norm.clamp(0.0, 1.0) } else { norm };
                });
            }
            return Ok(norm_arr);
        }
    }
}
