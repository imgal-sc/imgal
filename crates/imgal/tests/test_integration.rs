use imgal::distribution::normalized_gaussian;
use imgal::error::ImgalError;
use imgal::integration::{composite_simpson, midpoint, simpson};

const TOLERANCE: f64 = 1e-10;
const SIGMA: f64 = 2.0;
const BINS: usize = 512;
const WIDTH: f64 = 4.0;
const CENTER: f64 = 2.0;

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < TOLERANCE
}

/// Tests that `composite_simpson` returns the expected values for integrating
/// a normalized Gaussian distribution.
#[test]
fn integration_composite_simpson_expected_results() -> Result<(), ImgalError> {
    let gauss_arr = normalized_gaussian(SIGMA, BINS, WIDTH, CENTER, false);
    let result_par = composite_simpson(&gauss_arr, None, true)?;
    let result_seq = composite_simpson(&gauss_arr, None, false)?;
    assert!(approx_equal(result_par, 0.9986155934));
    assert!(approx_equal(result_seq, 0.9986155934));
    Ok(())
}

/// Tests that `midpoint` returns the expected values for integrating a
/// normalized Gaussian distribution.
#[test]
fn integration_midpoint_expected_results() {
    let gauss_arr = normalized_gaussian(SIGMA, BINS, WIDTH, CENTER, false);
    let result_par = midpoint(&gauss_arr, None, true);
    let result_seq = midpoint(&gauss_arr, None, false);
    assert!(approx_equal(result_par, 1.0));
    assert!(approx_equal(result_seq, 1.0));
}

/// Tests that `simpson` returns the expected values for integrating a
/// normalized Gaussian distribution.
#[test]
fn integration_simpson_expected_results() -> Result<(), ImgalError> {
    let gauss_arr = normalized_gaussian(SIGMA, 511, WIDTH, CENTER, false);
    let result_par = simpson(&gauss_arr, None, true)?;
    let result_seq = simpson(&gauss_arr, None, false)?;
    assert!(approx_equal(result_par, 0.9986128844));
    assert!(approx_equal(result_seq, 0.9986128844));
    Ok(())
}
