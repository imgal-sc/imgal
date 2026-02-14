use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr, Zip};

use crate::error::ImgalError;
use crate::statistics::min_max;
use crate::traits::numeric::AsNumeric;

/// Create an image histogram from an n-dimensional array.
///
/// # Description
///
/// Creates a 1-dimensional image histogram from an n-dimensional array.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array.
/// * `bins`: The number of bins to use for the image histogram. If `None`, then
///   `bins = 256`.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(Vec<i64>)`: The image histogram of the input n-dimensional array of size
///   `bins`. Each element represents the count of values falling into the
///   corresponding bin.
/// * `Err(ImgalError)`: If the input data array is empty or `bins == 0`.
pub fn histogram<'a, T, A, D>(
    data: A,
    bins: Option<usize>,
    parallel: bool,
) -> Result<Vec<i64>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let bins = bins.unwrap_or(256);
    if data.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" });
    }
    if bins == 0 {
        return Err(ImgalError::InvalidParameterValueEqual {
            param_name: "bins",
            value: 0,
        });
    }
    let (min, max) = min_max(&data, parallel)?;
    let bin_width: f64 = (max.to_f64() - min.to_f64()) / bins as f64;
    if parallel {
        let hist = Zip::from(&data).par_fold(
            || vec![0; bins],
            |mut thread_hist, &v| {
                let bin_index: usize = ((v.to_f64() - min.to_f64()) / bin_width) as usize;
                let bin_index = bin_index.min(bins - 1);
                thread_hist[bin_index] += 1;
                thread_hist
            },
            |mut hist_a, hist_b| {
                hist_a
                    .iter_mut()
                    .zip(hist_b.iter())
                    .for_each(|(a, b)| *a += b);
                hist_a
            },
        );
        Ok(hist)
    } else {
        let mut hist = vec![0; bins];
        data.iter().for_each(|&v| {
            let bin_index: usize = ((v.to_f64() - min.to_f64()) / bin_width) as usize;
            let bin_index = bin_index.min(bins - 1);
            hist[bin_index] += 1;
        });
        Ok(hist)
    }
}

/// Compute the histogram bin midpoint value from a bin index.
///
/// # Description
///
/// Computes the midpoint value of an image histogram bin at the given index.
/// The midpoint value is the center value of the bin range.
///
/// # Arguments
///
/// * `index`: The histogram bin index.
/// * `min`: The minimum value of the source data used to construct the
///   histogram.
/// * `max`: The maximum value of the source data used to construct the
///   histogram.
/// * `bins`: The number of bins in the histogram.
///
/// # Returns
///
/// * `Ok(T)`: The midpoint bin value of the specified index.
/// * `Err(ImgalError)`: If `bins == 0`.
#[inline]
pub fn histogram_bin_midpoint<T>(index: usize, min: T, max: T, bins: usize) -> Result<T, ImgalError>
where
    T: AsNumeric,
{
    if bins == 0 {
        return Err(ImgalError::InvalidParameterValueEqual {
            param_name: "bins",
            value: 0,
        });
    }
    let min = min.to_f64();
    let max = max.to_f64();
    let bin_width = (max - min) / bins as f64;
    Ok(T::from_f64(min + (index as f64 + 0.5) * bin_width))
}

/// Compute the histogram bin value range from a bin index.
///
/// # Description
///
/// Computes the start and end values (_i.e._ the range) for a specified
/// histogram bin index.
///
/// # Arguments
///
/// * `index`: The histogram bin index.
/// * `min`: The minimum value of the source data used to construct the
///   histogram.
/// * `max`: The maximum value of the source data used to construct the
///   histogram.
/// * `bins`: The number of bins in the histogram.
///
/// # Returns
///
/// * `Ok((T, T))`: A tuple containing the start and end values representing the
///   value range of the specified bin index.
/// * `Err(ImgalError)`: If `bins == 0`.
#[inline]
pub fn histogram_bin_range<T>(
    index: usize,
    min: T,
    max: T,
    bins: usize,
) -> Result<(T, T), ImgalError>
where
    T: AsNumeric,
{
    if bins == 0 {
        return Err(ImgalError::InvalidParameterValueEqual {
            param_name: "bins",
            value: 0,
        });
    }
    let min = min.to_f64();
    let max = max.to_f64();
    let bin_width = (max - min) / bins as f64;
    let bin_start = min + (index as f64 * bin_width);
    let bin_end = bin_start + bin_width;
    Ok((T::from_f64(bin_start), T::from_f64(bin_end)))
}
