use ndarray::{ArrayD, ArrayViewD};

use crate::traits::numeric::AsNumeric;

/// TODO
///
/// # Description
///
/// blah blah
///
/// # Arguments
///
/// * `data`:
/// * `min`:
/// * `max`:
/// * `clip`:
/// * `epsilon`:
///
/// # Returns
///
/// * `ArrayD<f64>`:
pub fn percentile_normalize<T>(
    data: ArrayViewD<T>,
    min: f64,
    max: f64,
    clip: Option<bool>,
    epsilon: Option<f64>,
) -> ArrayD<f64>
where
    T: AsNumeric,
{
    // set optional parameters if needed
    let clip = clip.unwrap_or(false);
    let epsilon = epsilon.unwrap_or(1e-20);

    // compute minumum and maximum percentile values from data
    todo!();
}
