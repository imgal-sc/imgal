use ndarray::{arr2, array, s};

use imgal::constants::RNG_SEED;
use imgal::integration::midpoint;
use imgal::prelude::*;
use imgal::simulation::blob::gaussian_metaballs;
use imgal::simulation::decay::{
    gaussian_exponential_decay_1d, gaussian_exponential_decay_3d, ideal_exponential_decay_1d,
    ideal_exponential_decay_3d, irf_exponential_decay_1d, irf_exponential_decay_3d,
};
use imgal::simulation::instrument::gaussian_irf_1d;
use imgal::simulation::noise::{poisson_noise, poisson_noise_mut};
use imgal::simulation::rng::Pcg;
use imgal::statistics::sum;

const TOLERANCE: f64 = 1e-10;
const SAMPLES: usize = 256;
const PERIOD: f64 = 12.5;
const TAUS: [f64; 2] = [1.0, 3.0];
const FRACTIONS: [f64; 2] = [0.7, 0.3];
const TOTAL_COUNTS: f64 = 5000.0;
const IRF_CENTER: f64 = 3.0;
const IRF_WIDTH: f64 = 0.5;
const SHAPE: (usize, usize) = (10, 10);
const CENTER: [[f64; 2]; 1] = [[25.0, 25.0]];
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const THREADS: Option<usize> = Some(0);

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < TOLERANCE
}

/// Tests that `gaussian_exponential_decay_1d` returns the expected photon count
/// total and values on the curve.
#[test]
fn decay_gaussian_exponential_decay_1d_expected_results() -> Result<(), ImgalError> {
    let data_par = gaussian_exponential_decay_1d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        THREADS,
    )?;
    let data_seq = gaussian_exponential_decay_1d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        None,
    )?;
    assert!(approx_equal(data_par[45], 0.0263672839));
    assert!(approx_equal(data_par[68], 135.7148429095));
    assert!(approx_equal(data_par[240], 1.3304021275));
    assert!(approx_equal(data_seq[45], 0.0263672839));
    assert!(approx_equal(data_seq[68], 135.7148429095));
    assert!(approx_equal(data_seq[240], 1.3304021275));
    assert!(approx_equal(sum(&data_par, None), 4960.5567668085,));
    assert!(approx_equal(sum(&data_seq, None), 4960.5567668085,));
    Ok(())
}

/// Tests that `gaussian_exponential_decay_3d` returns the expected photon count
/// total and values in the 3D array along the curve.
#[test]
fn decay_gaussian_exponential_decay_3d_expected_results() -> Result<(), ImgalError> {
    let data_par = gaussian_exponential_decay_3d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        SHAPE,
        THREADS,
    )?;
    let data_seq = gaussian_exponential_decay_3d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        SHAPE,
        THREADS,
    )?;
    assert_eq!(data_par.shape(), [10, 10, 256]);
    assert_eq!(data_seq.shape(), [10, 10, 256]);
    assert!(approx_equal(
        sum(data_par.slice(s![5, 5, ..]), None),
        4960.5567668085
    ));
    assert!(approx_equal(
        sum(data_seq.slice(s![5, 5, ..]), None),
        4960.5567668085
    ));
    assert!(approx_equal(data_par[[5, 5, 45]], 0.0263672839));
    assert!(approx_equal(data_par[[5, 5, 68]], 135.7148429095));
    assert!(approx_equal(data_par[[5, 5, 240]], 1.3304021275));
    assert!(approx_equal(data_seq[[5, 5, 45]], 0.0263672839));
    assert!(approx_equal(data_seq[[5, 5, 68]], 135.7148429095));
    assert!(approx_equal(data_seq[[5, 5, 240]], 1.3304021275));
    assert!(approx_equal(sum(&data_par, None), 496055.6766808581));
    assert!(approx_equal(sum(&data_seq, None), 496055.6766808581));
    Ok(())
}

/// Tests that `ideal_exponential_decay_1d` returns the expected photon count
/// total and values in the 1D array along the curve.
#[test]
fn decay_ideal_exponential_decay_1d_expected_results() -> Result<(), ImgalError> {
    let data_par =
        ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, THREADS)?;
    let data_seq =
        ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, None)?;
    assert!(approx_equal(data_par[10], 124.0242868016));
    assert!(approx_equal(data_par[30], 53.625382823));
    assert!(approx_equal(data_par[50], 25.2361154379));
    assert!(approx_equal(data_seq[10], 124.0242868016));
    assert!(approx_equal(data_seq[30], 53.625382823));
    assert!(approx_equal(data_seq[50], 25.2361154379));
    assert!(approx_equal(sum(&data_par, None), 5000.0));
    assert!(approx_equal(sum(&data_seq, None), 5000.0));
    Ok(())
}

/// Tests that `ideal_exponential_decay_3d` returns the expected photon count
/// total and values in the 3D array along the curve.
#[test]
fn decay_ideal_exponential_decay_3d_expected_results() -> Result<(), ImgalError> {
    let data_par = ideal_exponential_decay_3d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        SHAPE,
        THREADS,
    )?;
    let data_seq = ideal_exponential_decay_3d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        SHAPE,
        None,
    )?;
    assert_eq!(data_par.shape(), [10, 10, 256]);
    assert_eq!(data_seq.shape(), [10, 10, 256]);
    assert!(approx_equal(data_par[[5, 5, 10]], 124.0242868016));
    assert!(approx_equal(data_par[[5, 5, 30]], 53.625382823));
    assert!(approx_equal(data_par[[5, 5, 50]], 25.2361154379));
    assert!(approx_equal(data_seq[[5, 5, 10]], 124.0242868016));
    assert!(approx_equal(data_seq[[5, 5, 30]], 53.625382823));
    assert!(approx_equal(data_seq[[5, 5, 50]], 25.2361154379));
    assert!(approx_equal(
        sum(data_par.slice(s![5, 5, ..]), None),
        5000.0
    ));
    assert!(approx_equal(
        sum(data_seq.slice(s![5, 5, ..]), None),
        5000.0
    ));
    Ok(())
}

/// Tests that `irf_exponential_decay_1d` returns the expected photon count
/// total and values in the 1D array along the curve.
#[test]
fn decay_irf_exponential_decay_1d_expected_results() -> Result<(), ImgalError> {
    let irf = gaussian_irf_1d(SAMPLES, PERIOD, IRF_CENTER, IRF_WIDTH, None);
    let data_par = irf_exponential_decay_1d(
        &irf,
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        THREADS,
    )?;
    let data_seq =
        irf_exponential_decay_1d(&irf, SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, None)?;
    assert!(approx_equal(data_par[45], 0.0263672839));
    assert!(approx_equal(data_par[68], 135.7148429095));
    assert!(approx_equal(data_par[240], 1.3304021275));
    assert!(approx_equal(data_seq[45], 0.0263672839));
    assert!(approx_equal(data_seq[68], 135.7148429095));
    assert!(approx_equal(data_seq[240], 1.3304021275));
    assert!(approx_equal(sum(&data_par, None), 4960.5567668085));
    assert!(approx_equal(sum(&data_seq, None), 4960.5567668085));
    Ok(())
}

/// Tests that `irf_exponential_decay_3d` returns the expected photon count
/// total and values in the 3D array along the curve.
#[test]
fn decay_irf_exponential_decay_3d_expected_results() -> Result<(), ImgalError> {
    let irf = gaussian_irf_1d(SAMPLES, PERIOD, IRF_CENTER, IRF_WIDTH, None);
    let data_par = irf_exponential_decay_3d(
        &irf,
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        SHAPE,
        THREADS,
    )?;
    let data_seq = irf_exponential_decay_3d(
        &irf,
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        SHAPE,
        None,
    )?;
    assert_eq!(data_par.shape(), [10, 10, 256]);
    assert_eq!(data_seq.shape(), [10, 10, 256]);
    assert!(approx_equal(data_par[[5, 5, 20]], 3.9e-15));
    assert!(approx_equal(data_par[[5, 5, 30]], 1.1e-10));
    assert!(approx_equal(data_par[[5, 5, 50]], 1.2320652096));
    assert!(approx_equal(data_seq[[5, 5, 20]], 3.9e-15));
    assert!(approx_equal(data_seq[[5, 5, 30]], 1.1e-10));
    assert!(approx_equal(data_seq[[5, 5, 50]], 1.2320652096));
    assert!(approx_equal(
        sum(data_par.slice(s![5, 5, ..]), None),
        4960.5567668085
    ));
    assert!(approx_equal(
        sum(data_seq.slice(s![5, 5, ..]), None),
        4960.5567668085
    ));
    Ok(())
}

/// Tests that `gaussian_irf_1d` returns the expected IRF by checking points
/// along the curve and integrating the curve (midpoint).
#[test]
fn instrument_gaussian_irf_1d_expected_results() {
    let irf_par = gaussian_irf_1d(SAMPLES, PERIOD, IRF_CENTER, IRF_WIDTH, THREADS);
    let irf_seq = gaussian_irf_1d(SAMPLES, PERIOD, IRF_CENTER, IRF_WIDTH, None);
    let dt = PERIOD / SAMPLES as f64;
    assert!(approx_equal(
        midpoint(&irf_par, Some(dt), None),
        0.048828125
    ));
    assert!(approx_equal(
        midpoint(&irf_seq, Some(dt), None),
        0.048828125
    ));
    assert!(approx_equal(irf_par[42], 4.9861e-6));
    assert!(approx_equal(irf_par[62], 0.0905441712));
    assert!(approx_equal(irf_par[82], 9.058e-7));
    assert!(approx_equal(irf_seq[42], 4.9861e-6));
    assert!(approx_equal(irf_seq[62], 0.0905441712));
    assert!(approx_equal(irf_seq[82], 9.058e-7));
}

/// Tests that `poisson_noise` returns the expected input arrays with Poisson
/// noise applied. This test *only* tests the sequential output. The parallel
/// outputs are *not* reproducible because each thread forks the internal PCG
/// used, thus the nubmer of threads can change how many PCGs are used.
#[test]
fn noise_poisson_noise_expected_results() -> Result<(), ImgalError> {
    let scale = 0.8;
    let simple_data = vec![10.0, 15.2, 23.4, 39.0, 48.0, 53.7];
    let simple_data_exp = array!(4.0, 16.0, 14.0, 23.0, 30.0, 41.0);
    let image_data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &[50, 50],
        None,
    )?;
    let simple_data_pn = poisson_noise(&simple_data, scale, None, None);
    let image_data_pn = poisson_noise(&image_data, scale, None, None);
    assert_eq!(simple_data_pn, simple_data_exp);
    assert_eq!(image_data_pn[[30, 30]], 6.0);
    assert_eq!(image_data_pn[[45, 25]], 2.0);
    assert_eq!(image_data_pn[[10, 10]], 5.0);
    Ok(())
}

/// Tests that `poisson_noise_mut` mutates the input arrays with expected
/// Poisson noise applied. This test *only* tests the sequential output.
/// The parallel outputs are *not* reproducible because each thread forks the
/// internal PCG used, thus the nubmer of threads can change how many PCGs are
/// used.
#[test]
fn noise_poisson_noise_mut_expected_results() -> Result<(), ImgalError> {
    let scale = 0.8;
    let mut simple_data = array!(10.0, 15.2, 23.4, 39.0, 48.0, 53.7).into_dyn();
    let simple_data_exp = array!(4.0, 16.0, 14.0, 23.0, 30.0, 41.0).into_dyn();
    let mut image_data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &[50, 50],
        None,
    )?;
    poisson_noise_mut(simple_data.view_mut(), scale, None, None);
    poisson_noise_mut(image_data.view_mut(), scale, None, None);
    assert_eq!(simple_data, simple_data_exp);
    assert_eq!(image_data[[30, 30]], 6.0);
    assert_eq!(image_data[[45, 25]], 2.0);
    assert_eq!(image_data[[10, 10]], 5.0);
    Ok(())
}

/// Tests that the `Pcg` returns the expected random f32 and u32 numbers.
#[test]
fn rng_pcg_expected_results() -> Result<(), ImgalError> {
    let mut prng = Pcg::new(RNG_SEED);
    let rand_vals_f32: Vec<f32> = (0..10).map(|_| prng.next_f32()).collect();
    let rand_vals_u32: Vec<u32> = (0..10).map(|_| prng.next_u32()).collect();
    let rand_vals_u32_range = (0..10)
        .map(|_| prng.next_u32_range(20..50))
        .collect::<Result<Vec<u32>, ImgalError>>()?;
    let rand_vals_f32_exp: [f32; 10] = [
        0.062270045,
        0.3876549,
        0.397314,
        0.14715159,
        0.047530174,
        0.5564274,
        0.7872379,
        0.4368844,
        0.6955766,
        0.9106101,
    ];
    let rand_vals_u32_exp: [u32; 10] = [
        1882667393, 2179971700, 556780140, 1729229571, 3466107838, 3175703619, 3384978090,
        1490401167, 1951341877, 1261463854,
    ];
    let rand_vals_u32_range_exp: [u32; 10] = [24, 20, 49, 46, 40, 28, 27, 48, 24, 40];
    assert_eq!(rand_vals_f32, rand_vals_f32_exp);
    assert_eq!(rand_vals_u32, rand_vals_u32_exp);
    assert_eq!(rand_vals_u32_range, rand_vals_u32_range_exp);
    Ok(())
}
