use ndarray::{Array2, arr2};

use imgal::copy::{copy_into, copy_into_flat, duplicate};
use imgal::prelude::*;
use imgal::simulation::blob::gaussian_metaballs;
use imgal::statistics::min_max;

const CENTER: [[f64; 2]; 1] = [[25.0, 25.0]];
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const SHAPE: [usize; 2] = [50, 50];
const THREADS: Option<usize> = Some(0);

/// Tests that `duplicate` returns a copy of the input data.
#[test]
fn copy_duplicate_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
        None,
    )?;
    let dup_par = duplicate(&data, THREADS);
    let dup_seq = duplicate(&data, None);
    assert_eq!(&data, &dup_par);
    assert_eq!(&data, &dup_seq);
    Ok(())
}

/// Tests that `copy_into` copies data into a pre-allocated array of the same
/// dimensions and data type.
#[test]
fn copy_copy_into_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
        None,
    )?;
    let mut dup_par = Array2::<f64>::zeros((SHAPE[0], SHAPE[1])).into_dyn();
    let mut dup_seq = Array2::<f64>::zeros((SHAPE[0], SHAPE[1])).into_dyn();
    copy_into(&data, dup_par.view_mut(), THREADS)?;
    copy_into(&data, dup_seq.view_mut(), None)?;
    assert_eq!(&data, &dup_par);
    assert_eq!(&data, &dup_seq);
    Ok(())
}

/// Tests that `copy_into_flat` copies the input data into a flat 1D array.
#[test]
fn copy_copy_into_flat_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
        None,
    )?;
    let flat_par = copy_into_flat(&data, THREADS);
    let flat_seq = copy_into_flat(&data, None);
    assert_eq!(&data.len(), &flat_par.len());
    assert_eq!(&data.len(), &flat_seq.len());
    assert_eq!(min_max(&data, None)?, min_max(&flat_par, None)?);
    assert_eq!(min_max(&data, None)?, min_max(&flat_seq, None)?);
    Ok(())
}
