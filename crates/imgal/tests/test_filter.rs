use imgal::filter::{fft_convolve_1d, fft_deconvolve_1d};
use imgal::prelude::*;
use imgal::simulation::decay::{gaussian_exponential_decay_1d, ideal_exponential_decay_1d};
use imgal::simulation::instrument::gaussian_irf_1d;
use imgal::statistics::sum;

const TOLERANCE: f64 = 1e-10;
const SAMPLES: usize = 256;
const PERIOD: f64 = 12.5;
const TAUS: [f64; 2] = [1.0, 3.0];
const FRACTIONS: [f64; 2] = [0.7, 0.3];
const TOTAL_COUNTS: f64 = 5000.0;
const IRF_CENTER: f64 = 3.0;
const IRF_WIDTH: f64 = 0.5;
const THREADS: Option<usize> = Some(0);

fn approx_equal(a: f64, b: f64, tol: Option<f64>) -> bool {
    (a - b).abs() < tol.unwrap_or(TOLERANCE)
}

/// Tests that `fft_convolve_1d` returns the expected values for photon count,
/// and a point on the curve of an ideal bioexponential decay curve convolved
/// with a Gaussian IRF.
#[test]
fn filter_fft_convolve_1d_expected_results() -> Result<(), ImgalError> {
    let decay_arr =
        ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, None)?;
    let irf_arr = gaussian_irf_1d(SAMPLES, PERIOD, IRF_CENTER, IRF_WIDTH, None);
    let conv_par = fft_convolve_1d(&decay_arr, &irf_arr, THREADS);
    let conv_seq = fft_convolve_1d(&decay_arr, &irf_arr, None);
    assert!(approx_equal(sum(&conv_par, None), 4960.5567668085, None));
    assert!(approx_equal(sum(&conv_seq, None), 4960.5567668085, None));
    assert!(approx_equal(conv_par[68], 135.7148429095, None));
    assert!(approx_equal(conv_seq[68], 135.7148429095, None));
    Ok(())
}

/// Tests that `fft_deconvolve_1d` returns the expected values for array sum and
/// a point on the curve of the deconvolved result (the recovered simulated
/// IRF).
#[test]
fn filter_fft_deconvolve_1d_expected_results() -> Result<(), ImgalError> {
    let gauss_decay_arr = gaussian_exponential_decay_1d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        None,
    )?;
    let decay_arr =
        ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, None)?;
    let dconv_par = fft_deconvolve_1d(&gauss_decay_arr, &decay_arr, None, THREADS);
    let dconv_seq = fft_deconvolve_1d(&gauss_decay_arr, &decay_arr, None, None);
    assert!(approx_equal(sum(&dconv_par, None), 0.9999755326, None));
    assert!(approx_equal(sum(&dconv_seq, None), 0.9999755326, None));
    assert!(approx_equal(dconv_par[62], 0.090544374, None));
    assert!(approx_equal(dconv_seq[62], 0.090544374, None));
    Ok(())
}
