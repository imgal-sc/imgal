use ndarray::{Array2, Axis, s};

use imgal::parameter::omega;
use imgal::phasor::calibration::{
    calibrate_coords, calibrate_gs_image, calibrate_gs_image_mut, modulation_and_phase,
};
use imgal::phasor::plot::{gs_mask, gs_modulation, gs_phase, monoexponential_coords};
use imgal::phasor::time_domain::{gs_image, imaginary_coord, real_coord};
use imgal::prelude::*;
use imgal::simulation::decay::{gaussian_exponential_decay_3d, ideal_exponential_decay_1d};
use imgal::simulation::noise::poisson_noise_mut;

const TOLERANCE: f64 = 1e-10;
const SAMPLES: usize = 256;
const PERIOD: f64 = 12.5;
const TAUS: [f64; 2] = [1.0, 3.0];
const FRACTIONS: [f64; 2] = [0.7, 0.3];
const TOTAL_COUNTS: f64 = 5000.0;
const IRF_CENTER: f64 = 3.0;
const IRF_WIDTH: f64 = 0.5;
const SHAPE: (usize, usize) = (10, 10);
const MODULATION: f64 = 0.7;
const PHASE: f64 = -0.981;
const THREADS: Option<usize> = Some(0);

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < TOLERANCE
}

fn get_circle_mask(shape: (usize, usize), center: (isize, isize), radius: isize) -> Array2<bool> {
    let (row, col) = shape;
    let (cx, cy) = center;
    let r2 = radius * radius;
    let y_min = (cy - radius).max(0);
    let y_max = (cy + radius).min(row as isize - 1);
    let x_min = (cx - radius).max(0);
    let x_max = (cx + radius).min(col as isize - 1);
    let mut mask = Array2::<bool>::default(shape);
    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let dx = cx - x;
            let dy = cy - y;
            // use the squared distance formula for a quick circle mask
            if dx * dx + dy * dy <= r2 {
                mask[[y as usize, x as usize]] = true;
            }
        }
    }
    mask
}

/// Tests that `calibrate_coords` returns the expected calibrated G and S
/// values.
#[test]
fn calibration_calibrate_coords_expected_results() {
    let g = -0.37;
    let s = 0.68;
    let coords_cal = calibrate_coords(g, s, MODULATION, PHASE);
    assert!(approx_equal(coords_cal.0, 0.2515280246));
    assert!(approx_equal(coords_cal.1, 0.4799902632));
}

/// Tests that `calibrate_gs_image` returns the expected calibrated G/S values
/// in a new image array.
#[test]
fn calibration_calibrate_gs_image_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_exponential_decay_3d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        SHAPE,
        None,
    )?;
    let gs_arr = gs_image(data.view(), PERIOD, None, None, None, false)?;
    let cal_gs_arr = calibrate_gs_image(gs_arr.view(), MODULATION, PHASE, None, false);
    let g_mean = cal_gs_arr.index_axis(Axis(2), 0).mean().unwrap();
    let s_mean = cal_gs_arr.index_axis(Axis(2), 1).mean().unwrap();
    assert!(approx_equal(cal_gs_arr[[5, 5, 0]], 0.2536762376));
    assert!(approx_equal(cal_gs_arr[[5, 5, 1]], 0.4819949555));
    assert!(approx_equal(g_mean, 0.2536762376));
    assert!(approx_equal(s_mean, 0.4819949555));
    Ok(())
}

/// Tests that `calibrate_gs_image_mut` mutates the input data with the expected
/// G/S calibrated values.
#[test]
fn calibration_calibrate_gs_image_mut_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_exponential_decay_3d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        SHAPE,
        None,
    )?;
    let mut gs_arr = gs_image(data.view(), PERIOD, None, None, None, false)?;
    calibrate_gs_image_mut(gs_arr.view_mut(), MODULATION, PHASE, None, false);
    let g_mean = gs_arr.index_axis(Axis(2), 0).mean().unwrap();
    let s_mean = gs_arr.index_axis(Axis(2), 1).mean().unwrap();
    assert!(approx_equal(gs_arr[[5, 5, 0]], 0.2536762376));
    assert!(approx_equal(gs_arr[[5, 5, 1]], 0.4819949555));
    assert!(approx_equal(g_mean, 0.2536762376));
    assert!(approx_equal(s_mean, 0.4819949555));
    Ok(())
}

/// Tests that `modulation_and_phase` returns the expected modulation and phase
/// values for the given parameters.
#[test]
fn calibration_modulation_and_phase_expected_results() {
    let w = omega(PERIOD);
    let mod_phs = modulation_and_phase(-0.055, 0.59, 1.1, w);
    assert!(approx_equal(mod_phs.0, 1.4768757234));
    assert!(approx_equal(mod_phs.1, -1.1586655116));
}

/// Tests that `gs_mask` maps G and S coordinates back to the original input
/// image as a boolean mask.
#[test]
fn plot_gs_mask_expected_results() -> Result<(), ImgalError> {
    let mut data = gaussian_exponential_decay_3d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        (50, 50),
        None,
    )?;
    poisson_noise_mut(data.view_mut().into_dyn(), 0.3, None, false);
    let gs_arr = gs_image(data.view(), PERIOD, None, None, None, false)?;
    let g_coords = gs_arr.slice(s![25..30, 25..30, 0]).flatten().to_vec();
    let s_coords = gs_arr.slice(s![25..30, 25..30, 1]).flatten().to_vec();
    let mask = gs_mask(gs_arr.view(), &g_coords, &s_coords, None, false)?;
    assert_eq!(mask[[28, 28]], true);
    assert_eq!(mask[[5, 5]], false);
    Ok(())
}

/// Tests that `gs_modulation` returns the expected modulation for a G and S
/// pair.
#[test]
fn plot_gs_modulation_expected_results() {
    let m = gs_modulation(0.71, 0.43);
    assert!(approx_equal(m, 0.8300602387));
}

/// Tests that `gs_phase` returns the expected phase for a G and S pair.
#[test]
fn plot_gs_phase_expected_results() {
    let p = gs_phase(0.71, 0.43);
    assert!(approx_equal(p, 0.5445517081));
}

/// Tests that `monoexponential_coords` returns the expected G and S values for
/// the given tau and omega values.
#[test]
fn plot_monoexponential_coords_expected_results() {
    let w = omega(PERIOD);
    let coords = monoexponential_coords(1.1, w);
    assert!(approx_equal(coords.0, 0.765860473));
    assert!(approx_equal(coords.1, 0.4234598078));
}

/// Tests that `gs_image` returns the expected G/S phasor image by checking
/// points inside the image (with and without a mask) and the mean of each
/// channel.
#[test]
fn time_domain_gs_image_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_exponential_decay_3d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        (100, 100),
        None,
    )?;
    let mask = get_circle_mask((100, 100), (50, 50), 8);
    let gs_no_mask = gs_image(data.view(), PERIOD, None, None, None, false)?;
    let gs_with_mask = gs_image(data.view(), PERIOD, Some(mask.view()), None, None, false)?;
    let g_no_mask_view = gs_no_mask.index_axis(Axis(2), 0);
    let s_no_mask_view = gs_no_mask.index_axis(Axis(2), 1);
    let g_with_mask_view = gs_with_mask.index_axis(Axis(2), 0);
    let s_with_mask_view = gs_with_mask.index_axis(Axis(2), 1);
    assert!(approx_equal(g_no_mask_view.mean().unwrap(), -0.3706731273));
    assert!(approx_equal(s_no_mask_view.mean().unwrap(), 0.6841432489));
    assert!(approx_equal(g_with_mask_view[[45, 52]], -0.3706731273));
    assert!(approx_equal(s_with_mask_view[[45, 52]], 0.6841432489));
    assert_eq!(g_with_mask_view[[5, 8]], 0.0);
    assert_eq!(s_with_mask_view[[5, 8]], 0.0);
    Ok(())
}

/// Tests that `imaginary_coord` returns the expected imaginary (S) coordinate.
#[test]
fn time_domain_imaginary_coord_expected_results() -> Result<(), ImgalError> {
    let data = ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, None)?;
    let s_coord_par = imaginary_coord(&data, PERIOD, None, THREADS);
    let s_coord_seq = imaginary_coord(&data, PERIOD, None, None);
    assert!(approx_equal(s_coord_par, 0.410217863));
    assert!(approx_equal(s_coord_seq, 0.410217863));
    Ok(())
}

/// Tests that `real_coord` returns the expected real (G) coordinate.
#[test]
fn time_domain_real_coord_expected_results() -> Result<(), ImgalError> {
    let data = ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, None)?;
    let g_coord_par = real_coord(&data, PERIOD, None, THREADS);
    let g_coord_seq = real_coord(&data, PERIOD, None, None);
    assert!(approx_equal(g_coord_par, 0.660137605));
    assert!(approx_equal(g_coord_seq, 0.660137605));
    Ok(())
}
