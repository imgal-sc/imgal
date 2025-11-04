use std::cmp::Ordering;

use crate::error::ImgalError;
use crate::statistics::weighted_merge_sort_mut;
use crate::traits::numeric::AsNumeric;

/// Compute the weighted Kendall's Tau-b rank correlation coefficient.
///
/// # Description
///
/// This function calculates a weighted Kendall's Tau-b rank correlation
/// coefficient between two datasets. This implementation uses a weighted merge
/// sort to count discordant pairs (inversions), and applies tie corrections for
/// both variables to compute the final Tau-b coefficient. Here the weighted
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
/// - `n₀` = total weighted pairs = `(Σwᵢ)² - Σwᵢ²`
/// - `n₁` = weighted tie correction for first variable
/// - `n₂` = weighted tie correction for second variable
///
/// # Arguments
///
/// * `data_a`: The first dataset for correlation analysis. Must be the same
///    length as `data_b`.
/// * `data_b`: The second dataset for correlation analysis. Must be the same
///    length as `data_a`.
/// * `weights`: The associated weights for each observation pait. Must be the
///    same length as both input datasets.
///
/// # Returns
///
/// * `OK(f64)`: The weighted Kendall's Tau-b correlation coefficient, ranging
///    between -1.0 (negative correlation), 0.0 (no correlation) and 1.0
///    (positive correlation).
/// * `Err(ImgalError)`: If input array lengths do not match.
pub fn weighted_kendall_tau_b<T>(
    data_a: &[T],
    data_b: &[T],
    weights: &[f64],
) -> Result<f64, ImgalError>
where
    T: AsNumeric,
{
    // check array lengths match
    let dl = data_a.len();
    if dl != data_b.len() || dl != weights.len() {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_len: dl,
            b_arr_len: data_b.len().min(weights.len()),
        });
    }

    // can not compute a tau for less than 2 elements
    if dl < 2 {
        return Ok(0.0);
    }

    // rank the data and create paired data
    let (a_ranks, a_tie_corr) = rank_with_weights(data_a, weights);
    let (b_ranks, b_tie_corr) = rank_with_weights(data_b, weights);
    let mut rank_pairs: Vec<(i32, i32, usize)> = a_ranks
        .iter()
        .zip(b_ranks.iter())
        .enumerate()
        .map(|(i, (&a, &b))| (a, b, i))
        .collect();
    rank_pairs.sort_by_key(|&(a, _, _)| a);

    // extract b ranks in "a" sorted order and associated weights
    let mut b_sorted: Vec<i32> = Vec::with_capacity(dl);
    let mut w_sorted: Vec<f64> = Vec::with_capacity(dl);
    rank_pairs.iter().for_each(|&(_, b, i)| {
        b_sorted.push(b);
        w_sorted.push(weights[i]);
    });

    // count weighted inversions (i.e. swaps)
    let swaps = weighted_merge_sort_mut(&mut b_sorted, &mut w_sorted).unwrap();

    // calculate total possible weighted pairs
    let total_w: f64 = weights.iter().sum();
    let sum_w_sqr: f64 = weights.iter().map(|w| w.powi(2)).sum();
    let total_w_pairs = total_w.powi(2) - sum_w_sqr;

    // calculate tau-b with tie corrections, discordant pairs and swaps are the same
    let c_pairs = (total_w_pairs / 2.0) - swaps;
    let numer = c_pairs - swaps;
    // denom will become 0 or NaN if the total weighted pairs and tie correction
    // are close, this happens when one of the inputs has the same value in the
    // all or most of the array
    let denom = ((total_w_pairs - a_tie_corr) * (total_w_pairs - b_tie_corr)).sqrt();
    if denom != 0.0 && !denom.is_nan() {
        let tau = numer / denom;
        // clamp tau to meaningful range of -1.0 and 1.0
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

/// Rank data and associated weights with a Kendall Tau-b tie correction
fn rank_with_weights<T>(data: &[T], weights: &[f64]) -> (Vec<i32>, f64)
where
    T: AsNumeric,
{
    // create indicies sorted by values
    let dl = data.len();
    let mut indices: Vec<usize> = (0..dl).collect();
    indices.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap_or(Ordering::Equal));

    // set up rank parameters
    let mut ranks: Vec<i32> = vec![0; dl];
    let mut tie_corr = 0.0;
    let mut cur_rank = 1;
    let mut i = 0;

    while i < dl {
        let cur_val = data[indices[i]];
        let mut j = i;
        let mut tied_indices: Vec<usize> = Vec::new();
        // find all values tied with current value
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
            // factor of 2 for symmetric pairs
            tie_corr += 2.0 * tie_group_corr
        }
        cur_rank += group_size;
        i = j;
    }

    (ranks, tie_corr)
}
