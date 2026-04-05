use imgal::distribution::{inverse_normal_cdf, normalized_gaussian};
use imgal::error::ImgalError;
use imgal::integration::midpoint;

const TOLERANCE: f64 = 1e-10;

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < TOLERANCE
}

/// Tests that `inverse_normal_cdf` returns the expected values for known
/// probabilites and boundary cases.
#[test]
fn distribution_inverse_normal_cdf_expected_results() -> Result<(), ImgalError> {
    assert_eq!(inverse_normal_cdf(0.1)?, -1.2815515641401563);
    assert_eq!(inverse_normal_cdf(0.975)?, 1.959963986120195);
    assert_eq!(inverse_normal_cdf(0.5)?, 0.0);
    assert_eq!(inverse_normal_cdf(0.0)?, f64::NEG_INFINITY);
    assert_eq!(inverse_normal_cdf(1.0)?, f64::INFINITY);
    Ok(())
}

/// Tests that `normalized_gaussian` returns the expected results for index
/// `100` and the distribution integrates to approximately `1.0`.
#[test]
fn distribution_normalized_gaussian_expected_results() {
    let gauss_arr_a_par = normalized_gaussian(0.5, 256, 0.4, 2.0, true);
    let gauss_arr_b_par = normalized_gaussian(2.0, 256, 4.0, 2.0, true);
    let gauss_arr_a_seq = normalized_gaussian(0.5, 256, 0.4, 2.0, false);
    let gauss_arr_b_seq = normalized_gaussian(2.0, 256, 4.0, 2.0, false);
    assert!(approx_equal(midpoint(&gauss_arr_a_par, None, false), 1.0));
    assert!(approx_equal(midpoint(&gauss_arr_b_par, None, false), 1.0));
    assert!(approx_equal(midpoint(&gauss_arr_a_seq, None, false), 1.0));
    assert!(approx_equal(midpoint(&gauss_arr_b_seq, None, false), 1.0));
    assert!(approx_equal(gauss_arr_a_par[100], 0.0021260086));
    assert!(approx_equal(gauss_arr_b_par[100], 0.0044655072));
    assert!(approx_equal(gauss_arr_a_seq[100], 0.0021260086));
    assert!(approx_equal(gauss_arr_b_seq[100], 0.0044655072));
}
