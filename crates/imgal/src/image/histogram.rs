use ndarray::{Array1, ArrayBase, AsArray, Dimension, ViewRepr, Zip};
use rayon::prelude::*;

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
#[inline]
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
    let max_bin_idx = bins.saturating_sub(1) as f64;
    let (min, max) = min_max(&data, threads)?;
    let (min, max) = (min.to_f64(), max.to_f64());
    let inv_bin_width: f64 = bins as f64 / (max - min);
    let hist_op = |v: T| -> usize {
        let bin_idx = (v.to_f64() - min) * inv_bin_width;
        (bin_idx.max(0.0).min(max_bin_idx)) as usize
    };
    let hist_fold = |mut acc: Vec<i64>| {
        if let Some(s) = data.as_slice_memory_order() {
            unrolled_hist_op(s, acc.as_mut_slice(), min, inv_bin_width, max_bin_idx);
        } else {
            data.rows().into_iter().for_each(|r| {
                if let Some(s) = r.as_slice_memory_order() {
                    unrolled_hist_op(s, acc.as_mut_slice(), min, inv_bin_width, max_bin_idx);
                } else {
                    r.iter().for_each(|&v| {
                        acc[hist_op(v)] += 1;
                    })
                }
            })
        }
        acc
    };
    Ok(par!(threads,
    seq_exp: Array1::from_vec(hist_fold(vec![0_i64; bins])),
    par_exp: Array1::from_vec(Zip::from(data.rows())
        .into_par_iter()
        .fold_with(vec![0_i64; bins], |mut acc, (r,)| {
            if let Some(s) = r.as_slice_memory_order() {
                unrolled_hist_op(s, acc.as_mut_slice(), min, inv_bin_width, max_bin_idx);
            } else {
                r.iter().for_each(|&v| {
                    acc[hist_op(v)] += 1;
                })
            }
            acc
        })
        .reduce(|| vec![0_i64; bins],
            |mut hist_a, hist_b| {
                hist_a.iter_mut().zip(hist_b.iter()).for_each(|(a, b)| *a += b);
                hist_a
            }))))
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
/// Computes the start and end values (*i.e.* the range) for a specified
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

#[inline(always)]
fn unrolled_hist_op<T>(data: &[T], hist: &mut [i64], min: f64, inv_bin_width: f64, max_bin_idx: f64)
where
    T: AsNumeric,
{
    let hist_op = |v: T| -> usize {
        let bin_idx = (v.to_f64() - min) * inv_bin_width;
        (bin_idx.max(0.0).min(max_bin_idx)) as usize
    };
    let (chunks, remainder) = data.as_chunks::<8>();
    chunks.iter().for_each(|c| {
        let v0 = hist_op(c[0]);
        let v1 = hist_op(c[1]);
        let v2 = hist_op(c[2]);
        let v3 = hist_op(c[3]);
        let v4 = hist_op(c[4]);
        let v5 = hist_op(c[5]);
        let v6 = hist_op(c[6]);
        let v7 = hist_op(c[7]);
        hist[v0] += 1;
        hist[v1] += 1;
        hist[v2] += 1;
        hist[v3] += 1;
        hist[v4] += 1;
        hist[v5] += 1;
        hist[v6] += 1;
        hist[v7] += 1;
    });
    remainder.iter().for_each(|&v| hist[hist_op(v)] += 1);
}
