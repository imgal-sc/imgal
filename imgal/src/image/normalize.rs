use ndarray::{ArrayBase, ArrayD, AsArray, Dimension, ViewRepr, Zip};

use crate::statistics::linear_percentile;
use crate::traits::numeric::AsNumeric;

/// Normalize an n-dimensional array using percentile-based minimum and maximum.
///
/// # Description
///
/// Performs percentile-based normalization of an input n-dimensional array with
/// minimum and maximum percentage within the range of `0.0` to `100.0`.
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
) -> ArrayD<f64>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    // create a view of the data
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();

    // set optional parameters if needed
    let clip = clip.unwrap_or(false);
    let epsilon = epsilon.unwrap_or(1e-20);

    // compute minumum and maximum percentile values from flattened input data
    let per_min: f64 = linear_percentile(&view, min, None, None).unwrap()[0];
    let per_max: f64 = linear_percentile(&view, max, None, None).unwrap()[0];

    // normalize the input array
    let denom = per_max - per_min + epsilon;
    let mut norm_arr = ArrayD::<f64>::zeros(view.shape());
    Zip::from(view.into_dyn())
        .and(norm_arr.view_mut())
        .for_each(|v, n| {
            *n = (v.to_f64() - per_min) / denom;
        });

    // clip the normalized array to 0..1
    if clip {
        Zip::from(&mut norm_arr).for_each(|v| {
            *v = (*v).clamp(0.0, 1.0);
        })
    }

    norm_arr
}
