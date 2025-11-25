/// Compute the effective sample size (ESS) of a weighted sample set.
///
/// # Description
///
/// This function computes the effective sample size (ESS) of a weighted sample
/// set. Only the weights of the associated sample set are needed. The ESS is
/// defined as:
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
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    let mut sum_w = 0.0;
    let mut sum_sqr_w = 0.0;

    weights.iter().for_each(|w| {
        sum_w += w;
        sum_sqr_w += w.powi(2);
    });

    if sum_sqr_w == 0.0 {
        0.0
    } else {
        sum_w.powi(2) / sum_sqr_w
    }
}
