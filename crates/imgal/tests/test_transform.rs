use ndarray::arr2;

use imgal::prelude::*;
use imgal::simulation::blob::gaussian_metaballs;
use imgal::transform::pad::{constant_pad, reflect_pad, zero_pad};

const TOLERANCE: f64 = 1e-10;
const CENTER_2D: [[f64; 2]; 1] = [[25.0, 25.0]];
const CENTER_3D: [[f64; 3]; 1] = [[5.0, 25.0, 25.0]];
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const SHAPE_2D: [usize; 2] = [50, 50];
const SHAPE_3D: [usize; 3] = [10, 50, 50];
const PAD_CONFIG_2D: [usize; 2] = [5, 5];
const PAD_CONFIG_3D: [usize; 3] = [5, 5, 5];
const THREADS: Option<usize> = Some(0);

fn approx_equal(a: f64, b: f64, tol: Option<f64>) -> bool {
    (a - b).abs() < tol.unwrap_or(TOLERANCE)
}

/// Tests that `constant_pad` returns the expected constant value padded array
/// (2D and 3D) by checking the center for the maximum value and padded regions
/// for the constant value.
#[test]
fn pad_constant_pad_expected_results() -> Result<(), ImgalError> {
    let data_2d = gaussian_metaballs(
        &arr2(&CENTER_2D),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE_2D,
        None,
    )?;
    let data_3d = gaussian_metaballs(
        &arr2(&CENTER_3D),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE_3D,
        None,
    )?;
    let pad_2d_right_par = constant_pad(&data_2d, 3.2, &PAD_CONFIG_2D, Some(0), THREADS)?;
    let pad_3d_right_par = constant_pad(&data_3d, 3.2, &PAD_CONFIG_3D, Some(0), THREADS)?;
    let pad_2d_right_seq = constant_pad(&data_2d, 3.2, &PAD_CONFIG_2D, Some(0), None)?;
    let pad_3d_right_seq = constant_pad(&data_3d, 3.2, &PAD_CONFIG_3D, Some(0), None)?;
    let pad_2d_left_par = constant_pad(&data_2d, 3.2, &PAD_CONFIG_2D, Some(1), THREADS)?;
    let pad_3d_left_par = constant_pad(&data_3d, 3.2, &PAD_CONFIG_3D, Some(1), THREADS)?;
    let pad_2d_left_seq = constant_pad(&data_2d, 3.2, &PAD_CONFIG_2D, Some(1), None)?;
    let pad_3d_left_seq = constant_pad(&data_3d, 3.2, &PAD_CONFIG_3D, Some(1), None)?;
    let pad_2d_sym_par = constant_pad(&data_2d, 3.2, &PAD_CONFIG_2D, Some(2), THREADS)?;
    let pad_3d_sym_par = constant_pad(&data_3d, 3.2, &PAD_CONFIG_3D, Some(2), THREADS)?;
    let pad_2d_sym_seq = constant_pad(&data_2d, 3.2, &PAD_CONFIG_2D, Some(2), None)?;
    let pad_3d_sym_seq = constant_pad(&data_3d, 3.2, &PAD_CONFIG_3D, Some(2), None)?;
    assert_eq!(pad_2d_right_par.shape(), &[55, 55]);
    assert_eq!(pad_3d_right_par.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_right_seq.shape(), &[55, 55]);
    assert_eq!(pad_3d_right_seq.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_left_par.shape(), &[55, 55]);
    assert_eq!(pad_3d_left_par.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_left_seq.shape(), &[55, 55]);
    assert_eq!(pad_3d_left_seq.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_sym_par.shape(), &[60, 60]);
    assert_eq!(pad_3d_sym_par.shape(), &[20, 60, 60]);
    assert_eq!(pad_2d_sym_seq.shape(), &[60, 60]);
    assert_eq!(pad_3d_sym_seq.shape(), &[20, 60, 60]);
    // check the center
    assert_eq!(pad_2d_right_par[[25, 25]], 10.0);
    assert_eq!(pad_3d_right_par[[5, 25, 25]], 10.0);
    assert_eq!(pad_2d_right_seq[[25, 25]], 10.0);
    assert_eq!(pad_3d_right_seq[[5, 25, 25]], 10.0);
    assert_eq!(pad_2d_left_par[[30, 30]], 10.0);
    assert_eq!(pad_3d_left_par[[10, 30, 30]], 10.0);
    assert_eq!(pad_2d_left_seq[[30, 30]], 10.0);
    assert_eq!(pad_3d_left_seq[[10, 30, 30]], 10.0);
    assert_eq!(pad_2d_sym_par[[30, 30]], 10.0);
    assert_eq!(pad_3d_sym_par[[10, 30, 30]], 10.0);
    assert_eq!(pad_2d_sym_seq[[30, 30]], 10.0);
    assert_eq!(pad_3d_sym_seq[[10, 30, 30]], 10.0);
    // check right padding
    assert_eq!(pad_2d_right_par[[10, 53]], 3.2);
    assert_eq!(pad_3d_right_par[[7, 10, 53]], 3.2);
    assert_eq!(pad_2d_right_seq[[10, 53]], 3.2);
    assert_eq!(pad_3d_right_seq[[7, 10, 53]], 3.2);
    // check left padding
    assert_eq!(pad_2d_left_par[[10, 3]], 3.2);
    assert_eq!(pad_3d_left_par[[7, 10, 3]], 3.2);
    assert_eq!(pad_2d_left_seq[[10, 3]], 3.2);
    assert_eq!(pad_3d_left_seq[[7, 10, 3]], 3.2);
    // check symmetrical
    assert_eq!(pad_2d_sym_par[[10, 3]], 3.2);
    assert_eq!(pad_3d_sym_par[[7, 10, 3]], 3.2);
    assert_eq!(pad_2d_sym_seq[[10, 3]], 3.2);
    assert_eq!(pad_3d_sym_seq[[7, 10, 3]], 3.2);
    assert_eq!(pad_2d_sym_par[[10, 58]], 3.2);
    assert_eq!(pad_3d_sym_par[[7, 10, 58]], 3.2);
    assert_eq!(pad_2d_sym_seq[[10, 58]], 3.2);
    assert_eq!(pad_3d_sym_seq[[7, 10, 58]], 3.2);
    Ok(())
}

/// Tests that `reflect_pad` returns the expected reflected value padded array
/// (2D and 3D) by checking the center for the maximum value and padded regions
/// for the reflected value.
#[test]
fn pad_reflect_pad_expected_results() -> Result<(), ImgalError> {
    let data_2d = gaussian_metaballs(
        &arr2(&CENTER_2D),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE_2D,
        None,
    )?;
    let data_3d = gaussian_metaballs(
        &arr2(&CENTER_3D),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE_3D,
        None,
    )?;
    let pad_2d_right_par = reflect_pad(&data_2d, &PAD_CONFIG_2D, Some(0), THREADS)?;
    let pad_3d_right_par = reflect_pad(&data_3d, &PAD_CONFIG_3D, Some(0), THREADS)?;
    let pad_2d_right_seq = reflect_pad(&data_2d, &PAD_CONFIG_2D, Some(0), None)?;
    let pad_3d_right_seq = reflect_pad(&data_3d, &PAD_CONFIG_3D, Some(0), None)?;
    let pad_2d_left_par = reflect_pad(&data_2d, &PAD_CONFIG_2D, Some(1), THREADS)?;
    let pad_3d_left_par = reflect_pad(&data_3d, &PAD_CONFIG_3D, Some(1), THREADS)?;
    let pad_2d_left_seq = reflect_pad(&data_2d, &PAD_CONFIG_2D, Some(1), None)?;
    let pad_3d_left_seq = reflect_pad(&data_3d, &PAD_CONFIG_3D, Some(1), None)?;
    let pad_2d_sym_par = reflect_pad(&data_2d, &PAD_CONFIG_2D, Some(2), THREADS)?;
    let pad_3d_sym_par = reflect_pad(&data_3d, &PAD_CONFIG_3D, Some(2), THREADS)?;
    let pad_2d_sym_seq = reflect_pad(&data_2d, &PAD_CONFIG_2D, Some(2), None)?;
    let pad_3d_sym_seq = reflect_pad(&data_3d, &PAD_CONFIG_3D, Some(2), None)?;
    assert_eq!(pad_2d_right_par.shape(), &[55, 55]);
    assert_eq!(pad_3d_right_par.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_right_seq.shape(), &[55, 55]);
    assert_eq!(pad_3d_right_seq.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_left_par.shape(), &[55, 55]);
    assert_eq!(pad_3d_left_par.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_left_seq.shape(), &[55, 55]);
    assert_eq!(pad_3d_left_seq.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_sym_par.shape(), &[60, 60]);
    assert_eq!(pad_3d_sym_par.shape(), &[20, 60, 60]);
    assert_eq!(pad_2d_sym_seq.shape(), &[60, 60]);
    assert_eq!(pad_3d_sym_seq.shape(), &[20, 60, 60]);
    // check the center
    assert_eq!(pad_2d_right_par[[25, 25]], 10.0);
    assert_eq!(pad_3d_right_par[[5, 25, 25]], 10.0);
    assert_eq!(pad_2d_right_seq[[25, 25]], 10.0);
    assert_eq!(pad_3d_right_seq[[5, 25, 25]], 10.0);
    assert_eq!(pad_2d_left_par[[30, 30]], 10.0);
    assert_eq!(pad_3d_left_par[[10, 30, 30]], 10.0);
    assert_eq!(pad_2d_left_seq[[30, 30]], 10.0);
    assert_eq!(pad_3d_left_seq[[10, 30, 30]], 10.0);
    assert_eq!(pad_2d_sym_par[[30, 30]], 10.0);
    assert_eq!(pad_3d_sym_par[[10, 30, 30]], 10.0);
    assert_eq!(pad_2d_sym_seq[[30, 30]], 10.0);
    assert_eq!(pad_3d_sym_seq[[10, 30, 30]], 10.0);
    // check right padding
    assert!(approx_equal(pad_2d_right_par[[10, 53]], 4.5783336177, None));
    assert!(approx_equal(
        pad_3d_right_par[[7, 10, 53]],
        4.5554990835,
        None
    ));
    assert!(approx_equal(pad_2d_right_seq[[10, 53]], 4.5783336177, None));
    assert!(approx_equal(
        pad_3d_right_seq[[7, 10, 53]],
        4.5554990835,
        None
    ));
    // check left padding
    assert!(approx_equal(pad_2d_left_par[[10, 3]], 3.1309456796, None));
    assert!(approx_equal(pad_3d_left_par[[7, 10, 3]], 3.09591993, None));
    assert!(approx_equal(pad_2d_left_seq[[10, 3]], 3.1309456796, None));
    assert!(approx_equal(pad_3d_left_seq[[7, 10, 3]], 3.09591993, None));
    // check symmetrical
    assert!(approx_equal(pad_2d_sym_par[[10, 3]], 3.1309456796, None));
    assert!(approx_equal(pad_3d_sym_par[[7, 10, 3]], 3.09591993, None));
    assert!(approx_equal(pad_2d_sym_seq[[10, 3]], 3.1309456796, None));
    assert!(approx_equal(pad_3d_sym_seq[[7, 10, 3]], 3.09591993, None));
    assert!(approx_equal(pad_2d_sym_par[[10, 58]], 3.6787944117, None));
    assert!(approx_equal(
        pad_3d_sym_par[[7, 10, 58]],
        3.6376399027,
        None
    ));
    assert!(approx_equal(pad_2d_sym_seq[[10, 58]], 3.6787944117, None));
    assert!(approx_equal(
        pad_3d_sym_seq[[7, 10, 58]],
        3.6376399027,
        None
    ));
    Ok(())
}

/// Tests that `zero_pad` returns the expected zero padded array (2D and 3D) by
/// checking the center for the maximum value and padded regions for the zero
/// value.
#[test]
fn pad_zero_pad_expected_results() -> Result<(), ImgalError> {
    let data_2d = gaussian_metaballs(
        &arr2(&CENTER_2D),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE_2D,
        None,
    )?;
    let data_3d = gaussian_metaballs(
        &arr2(&CENTER_3D),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE_3D,
        None,
    )?;
    let pad_2d_right_par = zero_pad(&data_2d, &PAD_CONFIG_2D, Some(0), THREADS)?;
    let pad_3d_right_par = zero_pad(&data_3d, &PAD_CONFIG_3D, Some(0), THREADS)?;
    let pad_2d_right_seq = zero_pad(&data_2d, &PAD_CONFIG_2D, Some(0), None)?;
    let pad_3d_right_seq = zero_pad(&data_3d, &PAD_CONFIG_3D, Some(0), None)?;
    let pad_2d_left_par = zero_pad(&data_2d, &PAD_CONFIG_2D, Some(1), THREADS)?;
    let pad_3d_left_par = zero_pad(&data_3d, &PAD_CONFIG_3D, Some(1), THREADS)?;
    let pad_2d_left_seq = zero_pad(&data_2d, &PAD_CONFIG_2D, Some(1), None)?;
    let pad_3d_left_seq = zero_pad(&data_3d, &PAD_CONFIG_3D, Some(1), None)?;
    let pad_2d_sym_par = zero_pad(&data_2d, &PAD_CONFIG_2D, Some(2), THREADS)?;
    let pad_3d_sym_par = zero_pad(&data_3d, &PAD_CONFIG_3D, Some(2), THREADS)?;
    let pad_2d_sym_seq = zero_pad(&data_2d, &PAD_CONFIG_2D, Some(2), None)?;
    let pad_3d_sym_seq = zero_pad(&data_3d, &PAD_CONFIG_3D, Some(2), None)?;
    assert_eq!(pad_2d_right_par.shape(), &[55, 55]);
    assert_eq!(pad_3d_right_par.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_right_seq.shape(), &[55, 55]);
    assert_eq!(pad_3d_right_seq.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_left_par.shape(), &[55, 55]);
    assert_eq!(pad_3d_left_par.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_left_seq.shape(), &[55, 55]);
    assert_eq!(pad_3d_left_seq.shape(), &[15, 55, 55]);
    assert_eq!(pad_2d_sym_par.shape(), &[60, 60]);
    assert_eq!(pad_3d_sym_par.shape(), &[20, 60, 60]);
    assert_eq!(pad_2d_sym_seq.shape(), &[60, 60]);
    assert_eq!(pad_3d_sym_seq.shape(), &[20, 60, 60]);
    // check the center
    assert_eq!(pad_2d_right_par[[25, 25]], 10.0);
    assert_eq!(pad_3d_right_par[[5, 25, 25]], 10.0);
    assert_eq!(pad_2d_right_seq[[25, 25]], 10.0);
    assert_eq!(pad_3d_right_seq[[5, 25, 25]], 10.0);
    assert_eq!(pad_2d_left_par[[30, 30]], 10.0);
    assert_eq!(pad_3d_left_par[[10, 30, 30]], 10.0);
    assert_eq!(pad_2d_left_seq[[30, 30]], 10.0);
    assert_eq!(pad_3d_left_seq[[10, 30, 30]], 10.0);
    assert_eq!(pad_2d_sym_par[[30, 30]], 10.0);
    assert_eq!(pad_3d_sym_par[[10, 30, 30]], 10.0);
    assert_eq!(pad_2d_sym_seq[[30, 30]], 10.0);
    assert_eq!(pad_3d_sym_seq[[10, 30, 30]], 10.0);
    // check right padding
    assert_eq!(pad_2d_right_par[[10, 53]], 0.0);
    assert_eq!(pad_3d_right_par[[7, 10, 53]], 0.0);
    assert_eq!(pad_2d_right_seq[[10, 53]], 0.0);
    assert_eq!(pad_3d_right_seq[[7, 10, 53]], 0.0);
    // check left padding
    assert_eq!(pad_2d_left_par[[10, 3]], 0.0);
    assert_eq!(pad_3d_left_par[[7, 10, 3]], 0.0);
    assert_eq!(pad_2d_left_seq[[10, 3]], 0.0);
    assert_eq!(pad_3d_left_seq[[7, 10, 3]], 0.0);
    // check symmetrical
    assert_eq!(pad_2d_sym_par[[10, 3]], 0.0);
    assert_eq!(pad_3d_sym_par[[7, 10, 3]], 0.0);
    assert_eq!(pad_2d_sym_seq[[10, 3]], 0.0);
    assert_eq!(pad_3d_sym_seq[[7, 10, 3]], 0.0);
    assert_eq!(pad_2d_sym_par[[10, 58]], 0.0);
    assert_eq!(pad_3d_sym_par[[7, 10, 58]], 0.0);
    assert_eq!(pad_2d_sym_seq[[10, 58]], 0.0);
    assert_eq!(pad_3d_sym_seq[[7, 10, 58]], 0.0);
    Ok(())
}
