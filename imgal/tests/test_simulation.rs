use ndarray::s;

use imgal::integration::midpoint;
use imgal::simulation::{decay, instrument, noise};
use imgal::statistics::sum;

// simulated bioexponential decay parameters
const SAMPLES: usize = 256;
const PERIOD: f64 = 12.5;
const TAUS: [f64; 2] = [1.0, 3.0];
const FRACTIONS: [f64; 2] = [0.7, 0.3];
const TOTAL_COUNTS: f64 = 5000.0;
const IRF_CENTER: f64 = 3.0;
const IRF_WIDTH: f64 = 0.5;
const SHAPE: (usize, usize) = (10, 10);

// helper functions
fn ensure_within_tolerance(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

#[test]
fn decay_gaussian_exponential_decay_1d() {
    // simulate decay data
    let i = decay::gaussian_exponential_decay_1d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
    )
    .unwrap();

    // check curve photon count and a point on the curve (near max)
    assert!(ensure_within_tolerance(sum(&i), 4960.5567668085005, 1e-12));
    assert!(ensure_within_tolerance(i[68], 135.7148429095218, 1e-12));
}

// test the simulation::decay module
#[test]
fn decay_gaussian_exponential_decay_3d() {
    // simulate decay data
    let i = decay::gaussian_exponential_decay_3d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        SHAPE,
    )
    .unwrap();

    // check curve photon count and a point on the curve (near max)
    assert_eq!(i.shape(), [10, 10, 256]);
    assert!(ensure_within_tolerance(
        sum(i.slice(s![5, 5, ..]).as_slice().unwrap()),
        4960.5567668085005,
        1e-12
    ));
    assert!(ensure_within_tolerance(
        i[[5, 5, 68]],
        135.7148429095218,
        1e-12
    ));
}

#[test]
fn decay_ideal_exponential_decay_1d() {
    // simulate decay data
    let i = decay::ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS)
        .unwrap();

    // check curve photon count and a point on the curve
    assert!(ensure_within_tolerance(sum(&i), 5000.0, 1e-12));
    assert!(ensure_within_tolerance(i[30], 53.625382823015336, 1e-12));
}

#[test]
fn decay_ideal_exponential_decay_3d() {
    // simulate decay data
    let i =
        decay::ideal_exponential_decay_3d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, SHAPE)
            .unwrap();

    // check curve photon count and a point on the curve
    assert_eq!(i.shape(), [10, 10, 256]);
    assert!(ensure_within_tolerance(
        sum(i.slice(s![5, 5, ..]).as_slice().unwrap()),
        5000.0,
        1e-12
    ));
    assert!(ensure_within_tolerance(
        i[[5, 5, 30]],
        53.625382823015336,
        1e-12
    ));
}

#[test]
fn decay_irf_exponential_decay_1d() {
    // simulate IRF data to convolve decay data
    let irf = instrument::gaussian_irf_1d(SAMPLES, PERIOD, IRF_CENTER, IRF_WIDTH);
    let i = decay::irf_exponential_decay_1d(&irf, SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS)
        .unwrap();

    // check the curve by integration and a point
    assert!(ensure_within_tolerance(sum(&i), 4960.5567668085005, 1e-12));
    assert!(ensure_within_tolerance(i[68], 135.7148429095218, 1e-12));
}

#[test]
fn decay_irf_exponential_decay_3d() {
    // simulate IRF data to convolve decay data
    let irf = instrument::gaussian_irf_1d(SAMPLES, PERIOD, IRF_CENTER, IRF_WIDTH);
    let i = decay::irf_exponential_decay_3d(
        &irf,
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        SHAPE,
    )
    .unwrap();

    // check the curve by integration and a point
    assert_eq!(i.shape(), [10, 10, 256]);
    assert!(ensure_within_tolerance(
        sum(i.slice(s![5, 5, ..]).as_slice().unwrap()),
        4960.5567668085005,
        1e-12
    ));
    assert!(ensure_within_tolerance(
        i[[5, 5, 68]],
        135.7148429095218,
        1e-12
    ));
}

// test the simulation::instrument module
#[test]
fn instrument_gaussian_irf_1d() {
    // simulate IRF data
    let irf = instrument::gaussian_irf_1d(SAMPLES, PERIOD, IRF_CENTER, IRF_WIDTH);

    // bin width for integration check
    let dt = PERIOD / SAMPLES as f64;

    // check the curve by integration and a point
    assert!(ensure_within_tolerance(
        midpoint(&irf, Some(dt)),
        0.048828125,
        1e-12
    ));
    assert!(ensure_within_tolerance(irf[62], 0.09054417121965984, 1e-12));
}

// test the simulation::noise module
#[test]
fn noise_poisson_noise_1d() {
    // create test data
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let scale = 0.5;
    let seed = Some(42);

    // apply noise and test if deterministic with seed
    let result_a = noise::poisson_noise_1d(&data, scale, seed);
    let result_b = noise::poisson_noise_1d(&data, scale, seed);

    // apply noise and test if not equal with different seed
    let result_c = noise::poisson_noise_1d(&data, scale, Some(30));

    assert_eq!(result_a, result_b);
    assert_ne!(data, result_a);
    assert_ne!(result_a, result_c);
    assert!(result_a.iter().all(|&x| x >= 0.0));
}

#[test]
fn noise_poisson_noise_1d_mut() {
    // create test data
    let mut data_a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let data_b = data_a.clone();
    let scale = 0.5;
    let seed = Some(42);

    // mutate decay data with noise
    noise::poisson_noise_1d_mut(&mut data_a, scale, seed);

    assert_ne!(data_a, data_b);
    assert!(data_a.iter().all(|&x| x >= 0.0));
}

#[test]
fn noise_poisson_noise_3d() {
    // simulate decay data
    let i =
        decay::ideal_exponential_decay_3d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, SHAPE)
            .unwrap();
    let scale = 0.5;
    let seed = Some(42);

    // apply noise and test if deterministic with seed
    let result_a = noise::poisson_noise_3d(i.view(), scale, seed, None).unwrap();
    let result_b = noise::poisson_noise_3d(i.view(), scale, seed, None).unwrap();

    // apply noise and test if not equal with different seed
    let result_c = noise::poisson_noise_3d(i.view(), scale, Some(30), None).unwrap();

    assert_eq!(result_a.shape(), [10, 10, 256]);
    assert_eq!(result_a, result_b);
    assert_ne!(result_a, result_c);
    assert!(result_a.iter().all(|&x| x >= 0.0));
}

#[test]
fn noise_poisson_noise_3d_mut() {
    // simulate decay data
    let mut i_a =
        decay::ideal_exponential_decay_3d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, SHAPE)
            .unwrap();
    let i_b = i_a.clone();
    let scale = 0.5;
    let seed = Some(42);

    // mutate decay data with noise
    noise::poisson_noise_3d_mut(i_a.view_mut(), scale, seed, None);

    assert_ne!(i_a, i_b);
    assert!(i_a.iter().all(|&x| x >= 0.0));
}
