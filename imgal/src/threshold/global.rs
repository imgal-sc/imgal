use ndarray::{ArrayD, ArrayViewD};

use crate::image::{histogram, histogram_bin_midpoint};
use crate::statistics::min_max;
use crate::threshold::manual_mask;
use crate::traits::numeric::AsNumeric;

/// Create a boolean mask using Otsu's method.
///
/// # Description
///
/// This function implements Nobuyuki Otsu's automatic threshold method,
/// returning a boolean threshold mask. The Otsu threshold value used to create
/// the mask is calculated by maximizing the between-class variance of the
/// assumed bimodal distribution in the image histogram.
///
/// # Arguments
///
/// * `data`: An n-dimensional image or array.
/// * `bins`: The number of bins to use to construct the image histogram for
///   Otsu's method, default = 256.
///
/// # Returns
///
/// * `ArrayD<bool>`: A boolean array of the same shape as the input image
///   with pixels that are greater than the computed Otsu threshold value set
///   as `true` and pixels that are below the Otsu threshold value set as
///   `false`.
///
/// # Reference
///
/// <https://doi.org/10.1109/TSMC.1979.4310076>
pub fn otsu_mask<T>(data: ArrayViewD<T>, bins: Option<usize>) -> ArrayD<bool>
where
    T: AsNumeric,
{
    let threshold = otsu_value(data.view(), bins);
    manual_mask(data.view(), threshold)
}

/// Compute the image threshold with Otsu's method.
///
/// # Description
///
/// This function implements Nobuyuki Otsu's automatic image threshold method,
/// returning the Otsu threshold value. The Otsu threshold value is calculated
/// by maximizing the between-class variance of the assumed bimodal distribution
/// in the image histogram.
///
/// # Arguments
///
/// * `data`: An n-dimensonal image or array.
/// * `bins`: The number of bins to use to construct the image histogram for
///   Otsu's method, default = 256.
///
/// # Returns
///
/// * `T`: The Otsu threshold value.
///
/// # Reference
///
/// <https://doi.org/10.1109/TSMC.1979.4310076>
pub fn otsu_value<T>(data: ArrayViewD<T>, bins: Option<usize>) -> T
where
    T: AsNumeric,
{
    // get image histogram and initialize otsu values
    let hist = histogram(data.view(), bins);
    let dl = hist.len();
    let (min, max) = min_max(data.view());
    let mut bcv: f64 = 0.0;
    let mut bcv_max: f64 = 0.0;
    let mut hist_sum: f64 = 0.0;
    let mut hist_inten: f64 = 0.0;
    let mut inten_k: f64 = 0.0;
    let mut k_star: usize = 0;
    let mut n_k: f64 = 0.0;
    hist.iter().enumerate().for_each(|(i, &v)| {
        let v = v as f64;
        hist_sum += v;
        hist_inten += i as f64 * v;
    });

    // compute threshold, here "k" is the current threshold at index "i"
    hist.iter()
        .skip(1)
        .take(dl - 2)
        .enumerate()
        .for_each(|(i, &v)| {
            let idx = i + 1;
            let v = v as f64;
            inten_k += idx as f64 * v;
            n_k += v;
            let denom = n_k * (hist_sum - n_k);
            if denom != 0.0 {
                let num = (n_k / hist_sum) * hist_inten - inten_k;
                bcv = num.powi(2) / denom;
            } else {
                bcv = 0.0;
            }
            if bcv >= bcv_max {
                bcv_max = bcv;
                k_star = idx;
            }
        });

    histogram_bin_midpoint(k_star, min, max, bins.unwrap_or(256))
}
