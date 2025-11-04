use ndarray::{ArrayD, ArrayViewD, Zip};

use crate::image::histogram;
use crate::traits::numeric::ToFloat64;

/// TODO
pub fn otsu_mask<T>(data: ArrayViewD<T>) -> ArrayD<bool>
where
    T: ToFloat64,
{
    todo!();
}

/// TODO
///
/// # Description
///
/// Otsu's threshold method
///
/// # Arguments
///
/// * `data`:
/// * `bins`:
///
/// # Returns
///
/// * `T`:
pub fn otsu_value<T>(data: ArrayViewD<T>, bins: Option<usize>) -> T
where
    T: ToFloat64,
{
    // get image histogram and initialize otsu values
    let hist = histogram(data.view(), bins);
    let dl = hist.len();
    let mut bcv: f64 = 0.0;
    let mut bcv_max: f64 = 0.0;
    let mut hist_sum: i64 = 0;
    let mut hist_intensity: i64 = 0;
    let mut inensity_k: i64 = 0;
    let mut k: i64 = 0;
    let mut k_star: i64 = 0;
    let mut n_k: i64 = 0;
    hist.iter().enumerate().for_each(|(i, &v)| {
       hist_sum += v;
       hist_intensity += i as i64 * v;
    });

    // compute threshold
    hist.iter().skip(1).take(dl - 2).enumerate().for_each(|(i, &v)| {
        // do threshold compute
    });
    todo!();
}
