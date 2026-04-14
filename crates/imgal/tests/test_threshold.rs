use ndarray::arr2;

use imgal::error::ImgalError;
use imgal::simulation::blob::gaussian_metaballs;
use imgal::threshold::global::{otsu_mask, otsu_value};
use imgal::threshold::manual::manual_mask;

const TOLERANCE: f64 = 1e-10;
const CENTER: [[f64; 2]; 1] = [[25.0, 25.0]];
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const SHAPE: [usize; 2] = [50, 50];

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < TOLERANCE
}

/// Test that `manual_mask` returns the expected mask by checking its size and
/// points inside the mask.
#[test]
fn threshold_manual_mask_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
    )?;
    let mask_par = manual_mask(&data, 8.5, true);
    let mask_seq = manual_mask(&data, 8.5, false);
    let mask_par_size = mask_par
        .iter()
        .filter(|&&v| v != false)
        .fold(0, |acc, _| acc + 1);
    let mask_seq_size = mask_seq
        .iter()
        .filter(|&&v| v != false)
        .fold(0, |acc, _| acc + 1);
    assert_eq!(mask_par[[25, 25]], true);
    assert_eq!(mask_par[[5, 8]], false);
    assert_eq!(mask_par[[35, 20]], true);
    assert_eq!(mask_seq[[25, 25]], true);
    assert_eq!(mask_seq[[5, 8]], false);
    assert_eq!(mask_seq[[35, 20]], true);
    assert_eq!(mask_par_size, 421);
    assert_eq!(mask_seq_size, 421);
    Ok(())
}

/// Tests that `otsu_mask` returns the expected mask by checking its size and
/// points inside the mask.
#[test]
fn threshold_otsu_mask_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
    )?;
    let mask_par = otsu_mask(&data, None, true)?;
    let mask_seq = otsu_mask(&data, None, false)?;
    let mask_par_size = mask_par
        .iter()
        .filter(|&&v| v != false)
        .fold(0, |acc, _| acc + 1);
    let mask_seq_size = mask_seq
        .iter()
        .filter(|&&v| v != false)
        .fold(0, |acc, _| acc + 1);
    assert_eq!(mask_par[[25, 25]], true);
    assert_eq!(mask_par[[5, 8]], false);
    assert_eq!(mask_par[[43, 20]], true);
    assert_eq!(mask_seq[[25, 25]], true);
    assert_eq!(mask_seq[[5, 8]], false);
    assert_eq!(mask_seq[[43, 20]], true);
    assert_eq!(mask_par_size, 1101);
    assert_eq!(mask_seq_size, 1101);
    Ok(())
}

/// Tests that `otsu_value` returns the expected threshold values.
#[test]
fn threshold_otsu_value_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
    )?;
    let threshold_par = otsu_value(&data, None, true)?;
    let threshold_seq = otsu_value(&data, None, false)?;
    assert!(approx_equal(threshold_par, 6.4339888756));
    assert!(approx_equal(threshold_seq, 6.4339888756));
    Ok(())
}
