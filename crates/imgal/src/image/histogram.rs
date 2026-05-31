use ndarray::{Array1, ArrayBase, AsArray, Dimension, ViewRepr, Zip};

use crate::prelude::*;
use crate::statistics::min_max;

/// Create an image histogram from an n-dimensional image.
///
/// # Description
///
/// Creates a 1D image histogram from an n-dimensional image.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image.
/// * `bins`: The number of bins to use for the image histogram. If `None`, then
///   `bins = 256`.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Ok(Array1<i64>)`: The image histogram of the input n-dimensional image of
///   size `bins`. Each element represents the count of values falling into the
///   corresponding bin.
/// * `Err(ImgalError)`: If the input data array is empty or `bins == 0`.
pub fn histogram<'a, T, A, D>(
    data: A,
    bins: Option<usize>,
    threads: Option<usize>,
) -> Result<Array1<i64>, ImgalError>
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
    let (min, max) = min_max(&data, threads)?;
    let bin_width: f64 = (max.to_f64() - min.to_f64()) / bins as f64;
    let hist_seq = || {
        let mut hist = vec![0; bins];
        data.iter().for_each(|&v| {
            let bin_index: usize = ((v.to_f64() - min.to_f64()) / bin_width) as usize;
            let bin_index = bin_index.min(bins - 1);
            hist[bin_index] += 1;
        });
        Array1::from_vec(hist)
    };
    let hist_par = || {
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
        Array1::from_vec(hist)
    };
    Ok(par!(threads, seq_exp: hist_seq(), par_exp: hist_par()))
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
