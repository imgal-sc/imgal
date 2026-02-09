use ndarray::{Array, ArrayBase, AsArray, Dimension, ViewRepr};

use crate::error::ImgalError;
use crate::image::{histogram, histogram_bin_midpoint};
use crate::statistics::min_max;
use crate::threshold::manual_mask;
use crate::traits::numeric::AsNumeric;

/// Create a boolean mask using Otsu's method.
///
/// # Description
///
/// Creates a boolean mask using Nobuyuki Otsu's automatic threshold method. The
/// Otsu threshold value used to create the mask is calculated by maximizing the
/// between-class variance of the assumed bimodal distribution in the image
/// histogram.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image or array.
/// * `bins`: The number of bins to use to construct the image histogram for
///   Otsu's method. If `None`, then `bins = 256`.
///
/// # Returns
///
/// * `Ok(Array<bool, D>)`: A boolean array of the same shape as the input image
///   with pixels that are greater than the computed Otsu threshold value set
///   as `true` and pixels that are below the Otsu threshold value set as
///   `false`.
/// * `Err(ImgalError)`: If the input data array is empty or `bins == 0`.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Reference
///
/// <https://doi.org/10.1109/TSMC.1979.4310076>
pub fn otsu_mask<'a, T, A, D>(
    data: A,
    bins: Option<usize>,
    parallel: bool,
) -> Result<Array<bool, D>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let threshold = otsu_value(&data, bins)?;

    Ok(manual_mask(data, threshold, parallel))
}

/// Compute an image threshold with Otsu's method.
///
/// # Description
///
/// Calculates an image threshold value using Nobuyuki Otsu's automatic image
/// threshold method. The Otsu threshold value is calculated by maximizing the
/// between-class variance of the assumed bimodal distribution in the image
/// histogram.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image or array.
/// * `bins`: The number of bins to use to construct the image histogram for
///   Otsu's method. If `None`, the `bins = 256`.
///
/// # Returns
///
/// * `Ok(T)`: The Otsu threshold value.
/// * `Err(ImgalError)`: If the input data array is empty or `bins == 0`.
///
/// # Reference
///
/// <https://doi.org/10.1109/TSMC.1979.4310076>
pub fn otsu_value<'a, T, A, D>(data: A, bins: Option<usize>) -> Result<T, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let hist = histogram(&data, bins, false)?;
    let dl = hist.len();
    let (min, max) = min_max(data, false)?;
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
    hist.iter().take(dl - 1).enumerate().for_each(|(i, &v)| {
        let v = v as f64;
        inten_k += i as f64 * v;
        n_k += v;
        let denom = n_k * (hist_sum - n_k);
        if denom != 0.0 {
            let num = (n_k / hist_sum) * hist_inten - inten_k;
            bcv = (num * num) / denom;
        } else {
            bcv = 0.0;
        }
        if bcv >= bcv_max {
            bcv_max = bcv;
            k_star = i;
        }
    });

    Ok(histogram_bin_midpoint(
        k_star,
        min,
        max,
        bins.unwrap_or(256),
    )?)
}
