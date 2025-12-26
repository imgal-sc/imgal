use ndarray::{ArrayBase, ArrayD, AsArray, Dimension, ViewRepr, Zip};

use crate::error::ImgalError;
use crate::statistics::linear_percentile;
use crate::traits::numeric::AsNumeric;

/// Normalize an n-dimensional array using percentile-based minimum and maximum.
///
/// # Description
///
/// Performs percentile-based normalization of an input n-dimensional array with
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
/// * `data`: An n-dimensional array to normalize.
/// * `min`: The minimum percentage to normalize.
/// * `max`: The maximum percentage to normalize.
/// * `clip`: Boolean to indicate whether to clamp the normalized values to the
///   range `0.0` to `100.0`. If `None`, then `clip = false`.
/// * `epsilon`: A small positive value to avoid division by zero. If `None`,
///   then `epsilon = 1e-20`.
///
/// # Returns
///
/// * `ArrayD<f64>`: The percentile normalized n-dimensonal array.
pub fn percentile_normalize<'a, T, A, D>(
    data: A,
    min: f64,
    max: f64,
    clip: Option<bool>,
    epsilon: Option<f64>,
) -> Result<ArrayD<f64>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    // validate min/max range
    if min < 0.0 {
        return Err(ImgalError::InvalidParameterValueOutsideRange {
            param_name: "min",
            value: min,
            min: 0.0,
            max: 1.0,
        });
    }
    if max < 0.0 {
        return Err(ImgalError::InvalidParameterValueOutsideRange {
            param_name: "max",
            value: max,
            min: 0.0,
            max: 1.0,
        });
    }

    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let clip = clip.unwrap_or(false);
    let epsilon = epsilon.unwrap_or(1e-20);

    // compute the minimum and maximum linear percentile values from the input
    // data (flattened) and normalize
    let per_min: f64 = linear_percentile(&view, min, None, None).unwrap()[0];
    let per_max: f64 = linear_percentile(&view, max, None, None).unwrap()[0];
    let denom = per_max - per_min + epsilon;
    let mut norm_arr = ArrayD::<f64>::zeros(view.shape());
    Zip::from(view.into_dyn())
        .and(norm_arr.view_mut())
        .for_each(|v, n| {
            *n = (v.to_f64() - per_min) / denom;
        });
    if clip {
        Zip::from(&mut norm_arr).for_each(|v| {
            *v = (*v).clamp(0.0, 1.0);
        })
    }

    Ok(norm_arr)
}
