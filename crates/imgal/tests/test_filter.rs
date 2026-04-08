use imgal::error::ImgalError;
use imgal::filter::{fft_convolve_1d, fft_deconvolve_1d};
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

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < TOLERANCE
}

/// Tests that `fft_convolve_1d` returns the expected values for photon count,
/// and a point on the curve of an ideal bioexponential decay curve convolved
/// with a Gaussian IRF.
#[test]
fn filter_fft_convolve_1d_expected_results() -> Result<(), ImgalError> {
    let decay_arr = ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS)?;
    let irf_arr = gaussian_irf_1d(SAMPLES, PERIOD, IRF_CENTER, IRF_WIDTH);
    let conv_par = fft_convolve_1d(&decay_arr, &irf_arr, true);
    let conv_seq = fft_convolve_1d(&decay_arr, &irf_arr, false);
    assert!(approx_equal(sum(&conv_par, false), 4960.5567668085));
    assert!(approx_equal(sum(&conv_seq, false), 4960.5567668085));
    assert!(approx_equal(conv_par[68], 135.7148429095));
    assert!(approx_equal(conv_seq[68], 135.7148429095));
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
    )?;
    let decay_arr = ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS)?;
    let dconv_par = fft_deconvolve_1d(&gauss_decay_arr, &decay_arr, None, true);
    let dconv_seq = fft_deconvolve_1d(&gauss_decay_arr, &decay_arr, None, false);
    assert!(approx_equal(sum(&dconv_par, false), 0.9999755326));
    assert!(approx_equal(sum(&dconv_seq, false), 0.9999755326));
    assert!(approx_equal(dconv_par[62], 0.090544374));
    assert!(approx_equal(dconv_seq[62], 0.090544374));
    Ok(())
}
