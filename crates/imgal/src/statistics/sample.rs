use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr};

use crate::traits::numeric::AsNumeric;

/// Compute the effective sample size (ESS) of a weighted sample set.
///
/// # Description
///
/// Computes the effective sample size (ESS) of a weighted sample set. Only the
/// weights of the associated sample set are needed. The ESS is defined as:
///
/// ```text
/// ESS = (Σ wᵢ)² / Σ (wᵢ²)
/// ```
///
/// # Arguments
///
/// * `weights`: A slice of non-negative weights where each element represents
///   the weight of an associated sample.
///
/// # Returns
///
/// * `f64`: The effective number of independent samples.
pub fn effective_sample_size<'a, T, A, D>(weights: A) -> f64
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = weights.into();
    let mut sum_w = 0.0;
    let mut sum_sqr_w = 0.0;
    data.iter().for_each(|w| {
        let w = w.to_f64();
        sum_w += w;
        sum_sqr_w += w * w;
    });

    if sum_sqr_w == 0.0 {
        0.0
    } else {
        (sum_w * sum_w) / sum_sqr_w
    }
}
