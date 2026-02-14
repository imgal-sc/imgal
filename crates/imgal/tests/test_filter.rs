use imgal::filter;
use imgal::simulation::{decay, instrument};
use imgal::statistics::sum;

// simulated bioexponential decay parameters, unit is nanoseconds
const SAMPLES: usize = 256;
const PERIOD: f64 = 12.5;
const TAUS: [f64; 2] = [1.0, 3.0];
const FRACTIONS: [f64; 2] = [0.7, 0.3];
const TOTAL_COUNTS: f64 = 5000.0;
const IRF_CENTER: f64 = 3.0;
const IRF_WIDTH: f64 = 0.5;

// helper functions
fn ensure_within_tolerance(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

#[test]
fn filter_fft_convolve_1d() {
    // simulate two signals to convolve
    let a = decay::ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS)
        .unwrap();
    let b = instrument::gaussian_irf_1d(SAMPLES, PERIOD, IRF_CENTER, IRF_WIDTH);
    let conv_par = filter::fft_convolve_1d(&a, &b, true);
    let conv_seq = filter::fft_convolve_1d(&a, &b, false);

    // check curve photon count and a point on the curve (near max)
    assert!(ensure_within_tolerance(
        sum(&conv_par, false),
        4960.5567668085005,
        1e-12
    ));
    assert!(ensure_within_tolerance(
        conv_par[68],
        135.7148429095218,
        1e-12
    ));
    assert!(ensure_within_tolerance(
        sum(&conv_seq, false),
        4960.5567668085005,
        1e-12
    ));
    assert!(ensure_within_tolerance(
        conv_seq[68],
        135.7148429095218,
        1e-12
    ));
}

#[test]
fn filter_fft_deconvolve_1d() {
    // simulate two signals to deconvolve
    let a = decay::gaussian_exponential_decay_1d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
    )
    .unwrap();
    let b = decay::ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS)
        .unwrap();
    let dconv_par = filter::fft_deconvolve_1d(&a, &b, None, true);
    let dconv_seq = filter::fft_deconvolve_1d(&a, &b, None, false);

    // check curve photon count and a point on the curve (near max)
    assert!(ensure_within_tolerance(
        sum(&dconv_par, false),
        0.9999755326287557,
        1e-12
    ));
    assert!(ensure_within_tolerance(
        dconv_par[62],
        0.0905443740772156,
        1e-12
    ));
    assert!(ensure_within_tolerance(
        sum(&dconv_seq, false),
        0.9999755326287557,
        1e-12
    ));
    assert!(ensure_within_tolerance(
        dconv_seq[62],
        0.0905443740772156,
        1e-12
    ));
}
