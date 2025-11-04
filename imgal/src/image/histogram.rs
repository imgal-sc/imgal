use ndarray::ArrayViewD;

use crate::statistics::min_max;
use crate::traits::numeric::ToFloat64;

/// Compute the image histogram from an n-dimensional array.
///
/// # Description
///
/// This function computes an image (_i.e._ frequency) histogram for the values
/// in the input n-dimensional array.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array to construct the histogram from.
/// * `bins`: The number of bins to use for the histogram, default = 256.
///
/// # Returns
///
/// * `Vec<i64>`: The histogram of the input n-dimensional array of size `bins`.
///    Each element represents the count of values falling into the
///    corresponding bin.
pub fn histogram<T>(data: ArrayViewD<T>, bins: Option<usize>) -> Vec<i64>
where
    T: ToFloat64,
{
    let bins = bins.unwrap_or(256);

    // return an empty histogram if bins is zero or array is zero
    if data.is_empty() || bins == 0 {
        return vec![0; 1];
    }

    // get min and max values
    let (min, max) = min_max(data.view());

    // construct histogram
    let mut hist = vec![0; bins];
    let bin_width: f64 = (max.to_f64() - min.to_f64()) / bins as f64;
    data.iter().for_each(|&v| {
        let bin_index: usize = ((v.to_f64() - min.to_f64()) / bin_width) as usize;
        let bin_index = bin_index.min(bins - 1);
        hist[bin_index] += 1;
    });

    hist
}

/// Compute the histogram bin midpoint value from a bin index.
///
/// # Description
///
/// This function computes the midpoint value of an image histogram bin index.
/// The midpoint value is the center value of the bin range.
///
/// # Arguments
///
/// * `index`: The histogram bin index.
/// * `min`: The minimum value of the source data used to construct the
///    histogram.
/// * `max`: The maximum value of the source data used to construct the
///    histogram.
/// * `bins`: The number of bins in the histogram.
///
/// # Returns
///
/// * `T`: The midpoint bin value of the specified index.
#[inline]
pub fn histogram_bin_midpoint<T>(index: usize, min: T, max: T, bins: usize) -> T
where
    T: ToFloat64 + From<f64>,
{
    let bin_width = (max.to_f64() - min.to_f64()) / bins as f64;
    let bin_value = min.to_f64() + (index as f64 + 0.5) * bin_width;
    T::from(bin_value)
}

/// Compute the histogram bin value range from a bin index.
///
/// # Description
///
/// This function computes the start and end values (_i.e._ the range) for a
/// specified bin index.
///
/// # Arguments
///
/// * `index`: The histogram bin index.
/// * `min`: The minimum value of the source data used to construct the
///    histogram.
/// * `max`: The maximum value of the source data used to construct the
///    histogram.
/// * `bins`: The number of bins in the histogram.
///
/// # Returns
///
/// * `(T, T)`: A tuple containing the start and end values representing the
///    value range of the specified bin index.
#[inline]
pub fn histogram_bin_range<T>(index: usize, min: T, max: T, bins: usize) -> (T, T)
where
    T: ToFloat64 + From<f64>,
{
    let bin_width = (max.to_f64() - min.to_f64()) / bins as f64;
    let bin_start = min.to_f64() + (index as f64 * bin_width);
    let bin_end = bin_start + bin_width;
    (T::from(bin_start), T::from(bin_end))
}
