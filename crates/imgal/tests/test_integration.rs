use imgal::distribution::normalized_gaussian;
use imgal::integration::{composite_simpson, midpoint, simpson};
use imgal::prelude::*;

const TOLERANCE: f64 = 1e-10;
const SIGMA: f64 = 2.0;
const BINS: usize = 512;
const WIDTH: f64 = 4.0;
const CENTER: f64 = 2.0;
const THREADS: Option<usize> = Some(0);

fn approx_equal(a: f64, b: f64, tol: Option<f64>) -> bool {
    (a - b).abs() < tol.unwrap_or(TOLERANCE)
}

/// Tests that `composite_simpson` returns the expected values for integrating
/// a normalized Gaussian distribution.
#[test]
fn integration_composite_simpson_expected_results() {
    let gauss_arr = normalized_gaussian(SIGMA, BINS, WIDTH, CENTER, None);
    let result_par = composite_simpson(&gauss_arr, None, THREADS);
    let result_seq = composite_simpson(&gauss_arr, None, None);
    assert!(approx_equal(result_par, 0.9986155934, None));
    assert!(approx_equal(result_seq, 0.9986155934, None));
}

/// Tests that `midpoint` returns the expected values for integrating a
/// normalized Gaussian distribution.
#[test]
fn integration_midpoint_expected_results() {
    let gauss_arr = normalized_gaussian(SIGMA, BINS, WIDTH, CENTER, None);
    let result_par = midpoint(&gauss_arr, None, THREADS);
    let result_seq = midpoint(&gauss_arr, None, None);
    assert!(approx_equal(result_par, 1.0, None));
    assert!(approx_equal(result_seq, 1.0, None));
}

/// Tests that `simpson` returns the expected values for integrating a
/// normalized Gaussian distribution.
#[test]
fn integration_simpson_expected_results() -> Result<(), ImgalError> {
    let gauss_arr = normalized_gaussian(SIGMA, 511, WIDTH, CENTER, None);
    let result_par = simpson(&gauss_arr, None, THREADS)?;
    let result_seq = simpson(&gauss_arr, None, None)?;
    assert!(approx_equal(result_par, 0.9986128844, None));
    assert!(approx_equal(result_seq, 0.9986128844, None));
    Ok(())
}
