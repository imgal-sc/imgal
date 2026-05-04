use std::cmp::Ordering;

use ndarray::{ArrayBase, ArrayView1, AsArray, Ix1, ViewRepr, Zip};

use crate::error::ImgalError;
use crate::statistics::weighted_merge_sort_mut;
use crate::traits::numeric::AsNumeric;

/// Compute the Pearson correlation coefficient between two 1D arrays.
///
/// # Description
///
/// Computes the Pearson correlation coefficient, a measure of linear
/// correlation between two sets of 1D data.
///
/// Pearson's correlation coefficient is computed as:
///
/// ```text
/// r = Σ[(aᵢ - mean(a)) × (bᵢ - mean(b))] / √[Σ(aᵢ - mean(a))² × Σ(bᵢ - mean(b))²]
/// ```
///
/// # Arguments
///
/// * `data_a`: The first array for correlation analysis.
/// * `data_b`: The second array for correlation analysis.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(f64)`: Pearson's correlatoin coefficient ranging between `-1.0`
///   (perfect negative correlation), `0.0` (no correlation), and `1.0`
///   (perfect positive correlation).
/// * `Err(ImgalError)`: If `data_a.len() != data_b.len()`. If `data_a.len()` or
///   `data_b.len()` is <= 2.
pub fn pearson<'a, T, A>(data_a: A, data_b: A, parallel: bool) -> Result<f64, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let data_a: ArrayBase<ViewRepr<&'a T>, Ix1> = data_a.into();
    let data_b: ArrayBase<ViewRepr<&'a T>, Ix1> = data_b.into();
    if data_a.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "data_a",
        });
    }
    if data_b.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "data_b",
        });
    }
    let n = data_a.len();
    if n != data_b.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "data_a",
            a_arr_len: n,
            b_arr_name: "data_b",
            b_arr_len: data_b.len(),
        });
    }
    if n <= 2 {
        return Err(ImgalError::InvalidArrayLengthMinimum {
            arr_name: "data_a",
            arr_len: n,
            min_len: 3,
        });
    }
    let n = n as f64;
    let (sum_a, sum_b) = if parallel {
        Zip::from(data_a).and(data_b).par_fold(
            || (0.0, 0.0),
            |acc, &a, &b| (acc.0 + a.to_f64(), acc.1 + b.to_f64()),
            |acc, res| (acc.0 + res.0, acc.1 + res.1),
        )
    } else {
        Zip::from(data_a)
            .and(data_b)
            .fold((0.0, 0.0), |acc, &a, &b| {
                (acc.0 + a.to_f64(), acc.1 + b.to_f64())
            })
    };
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;
    let (numer, sq_a, sq_b) = if parallel {
        Zip::from(data_a).and(data_b).par_fold(
            || (0.0, 0.0, 0.0),
            |acc, &a, &b| {
                let diff_a = a.to_f64() - mean_a;
                let diff_b = b.to_f64() - mean_b;
                (
                    acc.0 + diff_a * diff_b,
                    acc.1 + diff_a * diff_a,
                    acc.2 + diff_b * diff_b,
                )
            },
            |acc, res| (acc.0 + res.0, acc.1 + res.1, acc.2 + res.2),
        )
    } else {
        Zip::from(data_a)
            .and(data_b)
            .fold((0.0, 0.0, 0.0), |acc, &a, &b| {
                let diff_a = a.to_f64() - mean_a;
                let diff_b = b.to_f64() - mean_b;
                (
                    acc.0 + diff_a * diff_b,
                    acc.1 + diff_a * diff_a,
                    acc.2 + diff_b * diff_b,
                )
            })
    };
    let denominator = (sq_a * sq_b).sqrt();
    if denominator == 0.0 {
        return Err(ImgalError::InvalidGeneric {
            msg: "Cannot compute Pearson correlation. One or both arrays have zero variance.",
        });
    }
    Ok(numer / denominator)
}

/// Compute the weighted Kendall's Tau-b rank correlation coefficient.
///
/// # Description
///
/// Calculates a weighted Kendall's Tau-b rank correlation coefficient between
/// two datasets. This implementation uses a weighted merge sort to count
/// discordant pairs (inversions), and applies tie corrections for both
/// variables to compute the final Tau-b coefficient. Here the weighted
/// observations contribute unequally to the final correlation coefficient.
///
/// The weighted Kendall's Tau-b is calculated using:
///
/// ```text
/// τ_b = (C - D) / √[(n₀ - n₁)(n₀ - n₂)]
/// ```
///
/// Where:
/// - `C` = number of weighted concordant pairs
/// - `D` = number of weighted discordant pairs
/// - `n₀` = total weighted pairs = `((Σwᵢ)² - Σwᵢ²) / 2.0`
/// - `n₁` = weighted tie correction for first variable
/// - `n₂` = weighted tie correction for second variable
///
/// # Arguments
///
/// * `data_a`: The first dataset for correlation analysis.
/// * `data_b`: The second dataset for correlation analysis.
/// * `weights`: The associated weights for each observation pait. Must be the
///   same length as both input datasets.
///
/// # Returns
///
/// * `OK(f64)`: The weighted Kendall's Tau-b correlation coefficient, ranging
///   between `-1.0` (perfect negative correlation), `0.0` (no correlation) and
///   `1.0` (perfect positive correlation).
/// * `Err(ImgalError)`: If `data_a.len() != data_b.len()`.
pub fn weighted_kendall_tau_b<'a, T, A, B>(
    data_a: A,
    data_b: A,
    weights: B,
) -> Result<f64, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    B: AsArray<'a, f64, Ix1>,
    T: 'a + AsNumeric,
{
    let data_a: ArrayBase<ViewRepr<&'a T>, Ix1> = data_a.into();
    let data_b: ArrayBase<ViewRepr<&'a T>, Ix1> = data_b.into();
    let weights: ArrayBase<ViewRepr<&'a f64>, Ix1> = weights.into();
    let dl = data_a.len();
    if dl != data_b.len() || dl != weights.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "data_a",
            a_arr_len: dl,
            b_arr_name: "data_b",
            b_arr_len: data_b.len().min(weights.len()),
        });
    }
    // can not compute a tau for less than 2 elements
    if dl < 2 {
        return Ok(0.0);
    }
    // kendall tau b is undefined if one or both data sets is uniform, here we
    // return NaN for this case
    let data_a_uniform = data_a.iter().all(|&v| v == data_a[0]);
    let data_b_uniform = data_b.iter().all(|&v| v == data_b[0]);
    if data_a_uniform || data_b_uniform {
        return Ok(f64::NAN);
    }
    // rank the input data arrays with weights, get "a" and "b" tie corrections
    let (a_ranks, a_tie_corr) = rank_with_weights(data_a, weights);
    let (b_ranks, b_tie_corr) = rank_with_weights(data_b, weights);
    let mut rank_pairs: Vec<(i32, i32, usize)> = a_ranks
        .iter()
        .zip(b_ranks.iter())
        .enumerate()
        .map(|(i, (&a, &b))| (a, b, i))
        .collect();
    // rank_pairs.sort_by_key(|&(a, _, _)| a);
    rank_pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    // extract "b" ranks in "a" sorted order and associated weights
    let mut b_sorted: Vec<i32> = Vec::with_capacity(dl);
    let mut w_sorted: Vec<f64> = Vec::with_capacity(dl);
    rank_pairs.iter().for_each(|&(_, b, i)| {
        b_sorted.push(b);
        w_sorted.push(weights[i]);
    });
    // calculate the disconcordant (D) pairs (i.e. "inversions" or "swaps") and
    // total weighted pairs as (Σwᵢ)² - Σwᵢ² which represents 2(C + D), where
    // (C) is the number of concordant pairs
    let swaps = weighted_merge_sort_mut(&mut b_sorted, &mut w_sorted).unwrap();
    let total_w: f64 = weights.iter().sum();
    let sum_w_sqr: f64 = weights.iter().map(|w| w * w).sum();
    let total_w_pairs = ((total_w * total_w) - sum_w_sqr) / 2.0;
    let c_pairs = total_w_pairs - swaps - a_tie_corr;
    let numer = c_pairs - swaps;
    // denom will become 0 or NaN if the total weighted pairs and tie correction
    // are close, this happens when one of the inputs has the same value in the
    // all or most of the array
    let denom = ((total_w_pairs - a_tie_corr) * (total_w_pairs - b_tie_corr)).sqrt();
    if denom != 0.0 && !denom.is_nan() {
        let tau = numer / denom;
        if tau >= 1.0 {
            Ok(1.0)
        } else if tau <= -1.0 {
            Ok(-1.0)
        } else {
            Ok(tau)
        }
    } else {
        Ok(0.0)
    }
}

/// Rank data and associated weights with a Kendall Tau-b tie correction.
fn rank_with_weights<T>(data: ArrayView1<T>, weights: ArrayView1<f64>) -> (Vec<i32>, f64)
where
    T: AsNumeric,
{
    let dl = data.len();
    let mut indices: Vec<usize> = (0..dl).collect();
    indices.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap_or(Ordering::Equal));
    let mut ranks: Vec<i32> = vec![0; dl];
    let mut tie_corr = 0.0;
    let mut cur_rank = 1;
    let mut i = 0;
    let mut tied_indices: Vec<usize> = Vec::new();
    while i < dl {
        let cur_val = data[indices[i]];
        let mut j = i;
        // find all values tied with current value
        tied_indices.clear();
        while j < dl && data[indices[j]].partial_cmp(&cur_val) == Some(Ordering::Equal) {
            tied_indices.push(indices[j]);
            j += 1;
        }
        // assign average rank to all tied values
        let group_size = (j - i) as i32;
        let avg_rank = cur_rank + (group_size - 1) / 2;
        tied_indices.iter().for_each(|&ti| {
            ranks[ti] = avg_rank;
        });
        // add tie corrections
        if group_size > 1 {
            let mut tie_group_corr = 0.0;
            for k in 0..tied_indices.len() {
                for l in (k + 1)..tied_indices.len() {
                    tie_group_corr += weights[tied_indices[k]] * weights[tied_indices[l]];
                }
            }
            tie_corr += tie_group_corr
        }
        cur_rank += group_size;
        i = j;
    }
    (ranks, tie_corr)
}
