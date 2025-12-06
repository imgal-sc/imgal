use crate::error::ImgalError;

// Acklam's rational approximations
const A: [f64; 6] = [
    -39.69683028665376,
    220.9460984245205,
    -275.9285104469687,
    138.3577518672690,
    -30.66479806614716,
    2.506628277459239,
];
const B: [f64; 5] = [
    -54.47609879822406,
    161.5858368580409,
    -155.6989798598866,
    66.80131188771972,
    -13.28068155288572,
];
const C: [f64; 6] = [
    -0.007784894002430293,
    -0.3223964580411365,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
];
const D: [f64; 4] = [
    0.007784695709041462,
    0.3224671290700398,
    2.445134137142996,
    3.754408661907416,
];

// Acklam's region thresholds
const P_LOW: f64 = 0.02425;
const P_HIGH: f64 = 1.0 - P_LOW;

/// Compute the quantile of a probability using the inverse normal cumulative
/// distribution function.
///
/// # Description
///
/// Computes the quantile (_z-score_) corresponding to a given cumulative
/// probabililty `prob` using Peter Acklam's rational approximation algorithm.
/// Acklam's algorithm has a relative error of less than `1.15e-9`.
///
/// # Arguments
///
/// * `prob`: The probability value in the range of `0.0` to `1.0`.
///
/// # Returns
///
/// * `Ok(f64)`: The quantile (z-score) corresponding to the given probability
///   `prob`.
/// * `Err(ImgalError)`: If the value of `prob` is less than `0.0` or greater
///   than `1.0`.
///
/// # Reference
///
/// <https://home.online.no/~pjacklam/notes/invnorm/>
pub fn inverse_normal_cdf(prob: f64) -> Result<f64, ImgalError> {
    // validate that "p" is within the valid range
    if prob < 0.0 || prob > 1.0 {
        return Err(ImgalError::InvalidParameterValueOutsideRange {
            param_name: "p",
            value: prob,
            min: 0.0,
            max: 1.0,
        });
    }

    // rational approximation for a lower region
    if prob < P_LOW {
        let q = (-2.0 * prob.ln()).sqrt();
        Ok(
            (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0),
        )
    // rational approximation for a central region
    } else if prob <= P_HIGH {
        let q = prob - 0.5;
        let r = q.powi(2);
        Ok(
            (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
                / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0),
        )
    // rational approximation for an upper region
    } else {
        let q = (-2.0 * (1.0 - prob).ln()).sqrt();
        Ok(
            -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0),
        )
    }
}
