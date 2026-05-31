use ndarray::arr2;

use imgal::image::{histogram, histogram_bin_midpoint, histogram_bin_range, percentile_normalize};
use imgal::prelude::*;
use imgal::simulation::blob::gaussian_metaballs;
use imgal::statistics::min_max;

const TOLERANCE: f64 = 1e-10;
const CENTER: [[f64; 2]; 1] = [[25.0, 25.0]];
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const SHAPE: [usize; 2] = [50, 50];
const THREADS: Option<usize> = Some(0);

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < TOLERANCE
}

/// Tests that `histogram` returns the expected values for the min/max of the
/// image histogram and values at the beginning, middle and end of the
/// histogram.
#[test]
fn image_histogram_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
        None,
    )?;
    let hist_par = histogram(&data, Some(256), THREADS)?;
    let hist_seq = histogram(&data, Some(256), None)?;
    let mm_par = min_max(&hist_par, None)?;
    let mm_seq = min_max(&hist_seq, None)?;
    assert_eq!(mm_par.0, 0);
    assert_eq!(mm_seq.0, 0);
    assert_eq!(mm_par.1, 32);
    assert_eq!(mm_seq.1, 32);
    assert_eq!(hist_par[0], 1);
    assert_eq!(hist_seq[0], 1);
    assert_eq!(hist_par[127], 16);
    assert_eq!(hist_seq[127], 16);
    assert_eq!(hist_par[255], 9);
    assert_eq!(hist_seq[255], 9);
    Ok(())
}

/// Tests that `histogram_bin_midpoint` returns the expected bin midpoint values
/// for both integer and floating point inputs.
#[test]
fn image_histogram_bin_midpoint_expected_results() -> Result<(), ImgalError> {
    assert_eq!(histogram_bin_midpoint(30, 0, 1200, 256)?, 142);
    assert_eq!(histogram_bin_midpoint(30, 0.0, 1200.0, 256)?, 142.96875);
    Ok(())
}

/// Tests that `histogram_bin_range` returns the expected start and end bin
/// value ranges (*i.e.* the range a given bin index represents) for integer and
/// floating point numbers.
#[test]
fn image_histogram_bin_range_expected_results() -> Result<(), ImgalError> {
    let (start_a, end_a) = histogram_bin_range(30, 0, 1200, 256)?;
    let (start_b, end_b) = histogram_bin_range(30, 0.0, 1200.0, 256)?;
    assert_eq!(start_a, 140);
    assert_eq!(end_a, 145);
    assert_eq!(start_b, 140.625);
    assert_eq!(end_b, 145.3125);
    Ok(())
}

/// Tests that `percentile_normalize` returns the expected values for per axis
/// and flat normalization with precentiles `1.0` and `99.8` (with and without
/// clipping).
#[test]
fn image_percentile_normalize_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
        None,
    )?;
    let flat_par = percentile_normalize(&data, 1.0, 99.8, false, None, None, THREADS)?;
    let flat_seq = percentile_normalize(&data, 1.0, 99.8, false, None, None, None)?;
    let flat_clip_par = percentile_normalize(&data, 1.0, 99.8, true, None, None, THREADS)?;
    let flat_clip_seq = percentile_normalize(&data, 1.0, 99.8, true, None, None, None)?;
    let ax_par = percentile_normalize(&data, 1.0, 99.8, false, Some(1), None, THREADS)?;
    let ax_seq = percentile_normalize(&data, 1.0, 99.8, false, Some(1), None, None)?;
    let ax_clip_par = percentile_normalize(&data, 1.0, 99.8, true, Some(1), None, THREADS)?;
    let ax_clip_seq = percentile_normalize(&data, 1.0, 99.8, true, Some(1), None, None)?;
    let (flat_min_par, flat_max_par) = min_max(&flat_par, None)?;
    let (flat_min_seq, flat_max_seq) = min_max(&flat_seq, None)?;
    let (ax_min_par, ax_max_par) = min_max(&ax_par, None)?;
    let (ax_min_seq, ax_max_seq) = min_max(&ax_seq, None)?;
    assert!(approx_equal(flat_par[[25, 25]], 1.0033992012));
    assert!(approx_equal(flat_seq[[25, 25]], 1.0033992012));
    assert!(approx_equal(flat_par[[36, 36]], 0.6476804184));
    assert!(approx_equal(flat_seq[[36, 36]], 0.6476804184));
    assert!(approx_equal(flat_par[[12, 12]], 0.5338065955));
    assert!(approx_equal(flat_seq[[12, 12]], 0.5338065955));
    assert!(approx_equal(flat_par[[10, 45]], 0.2645655818));
    assert!(approx_equal(flat_seq[[10, 45]], 0.2645655818));
    assert!(approx_equal(flat_par[[10, 43]], 0.3267436448));
    assert!(approx_equal(flat_seq[[10, 43]], 0.3267436448));
    assert!(approx_equal(ax_par[[25, 25]], 1.0002319179));
    assert!(approx_equal(ax_seq[[25, 25]], 1.0002319179));
    assert!(approx_equal(ax_par[[36, 36]], 0.7343214015));
    assert!(approx_equal(ax_seq[[36, 36]], 0.7343214015));
    assert!(approx_equal(ax_par[[12, 12]], 0.6394860449));
    assert!(approx_equal(ax_seq[[12, 12]], 0.6394860449));
    assert!(approx_equal(ax_par[[10, 45]], 0.5358021143));
    assert!(approx_equal(ax_seq[[10, 45]], 0.5358021143));
    assert!(approx_equal(ax_par[[10, 43]], 0.5358021143));
    assert!(approx_equal(ax_seq[[10, 43]], 0.5358021143));
    assert!(approx_equal(flat_min_par, -0.0736970979));
    assert!(approx_equal(flat_min_seq, -0.0736970979));
    assert!(approx_equal(flat_max_par, 1.00339920120));
    assert!(approx_equal(flat_max_seq, 1.00339920120));
    assert!(approx_equal(ax_min_par, -0.0268440183));
    assert!(approx_equal(ax_min_seq, -0.0268440183));
    assert!(approx_equal(ax_max_par, 1.00023191799));
    assert!(approx_equal(ax_max_seq, 1.00023191799));
    assert_eq!(min_max(&flat_clip_par, None)?, (0.0, 1.0));
    assert_eq!(min_max(&flat_clip_seq, None)?, (0.0, 1.0));
    assert_eq!(min_max(&ax_clip_par, None)?, (0.0, 1.0));
    assert_eq!(min_max(&ax_clip_seq, None)?, (0.0, 1.0));
    Ok(())
}
